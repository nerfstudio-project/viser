"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from .utils import Rays

_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def _load_colmap(root_fp: str, subject_id: str, factor: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0/")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # # Switch from COLMAP (right, down, fwd) to Nerf (right, up, back) frame.
    # poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    assert params is None, "Only support pinhole camera model."

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    if factor > 1:
        image_dir_suffix = f"_{factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]
    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % 8 == 0],
        "train": all_indices[all_indices % 8 != 0],
    }
    return images, camtoworlds, K, split_indices


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]
    SUBJECT_IDS = [
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
    ]

    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.K, split_indices = _load_colmap(
            root_fp, subject_id, factor
        )
        # normalize the scene
        T, sscale = similarity_from_cameras(
            self.camtoworlds, strict_scaling=False
        )
        self.camtoworlds = np.einsum("nij, ki -> nkj", self.camtoworlds, T)
        self.camtoworlds[:, :3, 3] *= sscale
        # split
        indices = split_indices[split]
        self.images = self.images[indices]
        self.camtoworlds = self.camtoworlds[indices]
        # to tensor
        self.images = torch.from_numpy(self.images).to(torch.uint8).to(device)
        self.camtoworlds = (
            torch.from_numpy(self.camtoworlds).to(torch.float32).to(device)
        )
        self.K = torch.tensor(self.K).to(torch.float32).to(device)
        self.height, self.width = self.images.shape[1:3]

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [num_rays, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
