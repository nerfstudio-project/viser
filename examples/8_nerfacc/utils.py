"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from nerfacc import OccupancyGrid, ray_marching, rendering
from nerfacc.proposal import rendering as rendering_proposal

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]
MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enlarge_aabb(aabb, factor: float) -> torch.Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def render_image_proposal(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    proposal_annealing: float = 1.0,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return proposal_network(positions)

    def rgb_sigma_fn(t_starts, t_ends):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-2], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        (
            rgb,
            opacity,
            depth,
            (weights_per_level, s_vals_per_level),
        ) = rendering_proposal(
            rgb_sigma_fn=rgb_sigma_fn,
            num_samples=num_samples,
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            num_samples_per_prop=num_samples_per_prop,
            rays_o=chunk_rays.origins,
            rays_d=chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=radiance_field.training,
            sampling_type=sampling_type,
            opaque_bkgd=opaque_bkgd,
            render_bkgd=render_bkgd,
            proposal_requires_grad=proposal_requires_grad,
            proposal_annealing=proposal_annealing,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        weights_per_level,
        s_vals_per_level,
    )
