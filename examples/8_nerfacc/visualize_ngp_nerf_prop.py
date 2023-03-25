"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import itertools
import pathlib
import time
from typing import Tuple

import imageio
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from nerfacc.proposal import (
    compute_prop_loss,
    get_proposal_annealing_fn,
    get_proposal_requires_grad_fn,
)
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from scipy.spatial.transform import Rotation
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    Rays,
    render_image_proposal,
    set_random_seed,
)

import viser


def quat_from_mat3(
    mat3: npt.NDArray[np.float32],
) -> Tuple[float, float, float, float]:
    return tuple(np.roll(Rotation.from_matrix(mat3).as_quat(), 1).tolist())


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = 0.0
    # scene parameters
    unbounded = True
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2  # TODO: Try 0.02
    far_plane = 1e3
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=256,
        ),
    ]
    # render parameters
    num_samples = 48
    num_samples_per_prop = [256, 96]
    sampling_type = "lindisp"
    opaque_bkgd = True

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = 1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    # scene parameters
    unbounded = False
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    sampling_type = "uniform"
    opaque_bkgd = False

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

checkpoint_path = (
    pathlib.Path(__file__).parent.parent
    / "assets"
    / "nerfacc"
    / f"{args.scene}_ngp_nerf_prop.pth"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# setup the radiance field we want to train.
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
radiance_field.load_state_dict(checkpoint["radiance_field"])
radiance_field.to(device)
for net, net_state_dict in zip(proposal_networks, checkpoint["proposal_networks"]):
    net.load_state_dict(net_state_dict)
    net.to(device)


server = viser.ViserServer()
server.reset_scene()

train_intrin = train_dataset.K.cpu().numpy()
train_c2ws = train_dataset.camtoworlds.cpu().numpy()
if train_dataset.OPENGL_CAMERA:
    train_c2ws = (train_c2ws @ np.diag([1, -1, -1, 1])).astype(np.float32)
train_img_wh = tuple(train_dataset.images.shape)[2:0:-1]

worldup_axis = train_c2ws[:, :3, 1].mean(0) * (1 if train_dataset.OPENGL_CAMERA else -1)
worldup_axis /= np.linalg.norm(worldup_axis)
# +Z is up by default.
a = np.cross(worldup_axis, np.array([0, 0, 1]))
norma = np.linalg.norm(a)
omega = np.arcsin(norma)
worldup_fix_mat3 = Rotation.from_rotvec(
    a * omega / np.clip(norma, 1e-7, None)
).as_matrix()


server.add_frame(
    f"/frame",
    wxyz=quat_from_mat3(worldup_fix_mat3),
    position=(0.0, 0.0, 0.0),
    axes_length=0.05,
    axes_radius=0.005,
)
for i, c2w in enumerate(train_c2ws):
    server.add_frame(
        f"/frame/train/{i}/camera",
        wxyz=quat_from_mat3(c2w[:3, :3]),
        position=tuple(c2w[:3, 3].tolist()),
        axes_length=0.05,
        axes_radius=0.005,
    )
    server.add_camera_frustum(
        f"/frame/train/{i}/camera/frustum",
        fov=2 * np.arctan2(train_img_wh[0] / 2, train_intrin[0, 0]),
        aspect=train_img_wh[0] / train_img_wh[1],
        scale=0.03,
    )


__import__("ipdb").set_trace()


@torch.inference_mode()
def render_image_from_camera(camera: viser.CameraState) -> None:
    # TODO(hangg): Hard-code for now. Should expose it to GUI and/or deternine
    # it from the rendering performance.
    W = 256
    H = int(W / camera.aspect)
    focal_length = W / (2 * np.tan(camera.fov / 2))
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(W, dtype=torch.float32, device=device),
                torch.arange(H, dtype=torch.float32, device=device),
                indexing="xy",
            ),
            2,
        )
        + 0.5
    )
    viewdirs = F.normalize(
        F.pad(
            torch.stack(
                [
                    (grid[..., 0] - W / 2) / focal_length,
                    (grid[..., 1] - H / 2) / focal_length,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        ),
        dim=-1,
    )
    c2w = (
        np.concatenate(
            [
                Rotation.from_quat(
                    (camera.wxyz[1], camera.wxyz[2], camera.wxyz[3], camera.wxyz[0])
                ).as_matrix(),
                np.array(camera.position)[:, None],
            ],
            axis=-1,
        )
    ).astype(np.float32)
    c2w = (worldup_fix_mat3.T @ c2w).astype(np.float32)
    viewdirs = (viewdirs[..., None, :] * torch.from_numpy(c2w[:3, :3]).to(device)).sum(
        dim=-1
    )
    origins = torch.broadcast_to(
        torch.tensor(c2w[:3, -1], dtype=torch.float32, device=device),
        viewdirs.shape,
    )
    rays = Rays(origins, viewdirs)

    # rendering
    (
        rgb,
        acc,
        depth,
        _,
        _,
    ) = render_image_proposal(
        radiance_field,
        proposal_networks,
        rays,
        scene_aabb=None,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=1.0,
        # test options
        test_chunk_size=args.test_chunk_size,
    )
    return rgb.cpu().numpy()


seen_clients = set()
while True:
    # Get all currently connected clients.
    clients = server.get_clients()
    print("Connected client IDs", clients.keys())

    for id, client in clients.items():
        # New client? We can attach a callback.
        if id not in seen_clients:
            seen_clients.add(id)

            # This will run whenever we get a new camera!
            @client.on_camera_update
            def camera_update(client: viser.ClientHandle) -> None:
                camera = client.get_camera()
                print("New camera", camera)
                server.set_background_image(render_image_from_camera(camera))

            # Show the client ID in the GUI.
            client.add_gui_text("Info", initial_value=f"Client {id}", disabled=True)

        camera = client.get_camera()
        server.set_background_image(render_image_from_camera(camera))
        print(f"Camera pose for client {id}")
        print(f"\twxyz: {camera.wxyz}")
        print(f"\tposition: {camera.position}")
        print(f"\tfov: {camera.fov}")
        print(f"\taspect: {camera.aspect}")
        print(f"\tlast update: {camera.last_updated}")

    time.sleep(0.1)
