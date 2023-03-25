import argparse
import pathlib
import time
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from nerfacc import OccupancyGrid
from radiance_fields.ngp import NGPRadianceField
from scipy.spatial.transform import Rotation
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    Rays,
    enlarge_aabb,
    render_image,
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
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.02
    far_plane = None
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = None
    far_plane = None
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=None,
    device=device,
    **train_dataset_kwargs,
)

# setup scene aabb
scene_aabb = enlarge_aabb(aabb, 1 << (grid_nlvl - 1))

checkpoint_path = (
    pathlib.Path(__file__).parent.parent
    / "assets"
    / "nerfacc"
    / f"{args.scene}_ngp_nerf.pth"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# setup the radiance field we want to visualize.
radiance_field = NGPRadianceField(aabb=scene_aabb)
radiance_field.load_state_dict(checkpoint["radiance_field"])
radiance_field.to(device)
occupancy_grid = OccupancyGrid(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
)
occupancy_grid.load_state_dict(checkpoint["occupancy_grid"])
occupancy_grid.to(device)

server = viser.ViserServer()
server.reset_scene()

train_intrin = train_dataset.K.cpu().numpy()
train_c2ws = train_dataset.camtoworlds.cpu().numpy()
train_c2ws = (train_c2ws @ np.diag([1, -1, -1, 1])).astype(np.float32)
train_img_wh = tuple(train_dataset.images.shape)[2:0:-1]
for i, c2w in enumerate(train_c2ws):
    server.add_frame(
        f"/train/{i}/camera",
        wxyz=quat_from_mat3(c2w[:3, :3]),
        position=tuple(c2w[:3, 3].tolist()),
        axes_length=0.1,
        axes_radius=0.01,
    )
    server.add_camera_frustum(
        f"/train/{i}/camera/frustum",
        fov=2 * np.arctan2(train_img_wh[0] / 2, train_intrin[0, 0]),
        aspect=train_img_wh[0] / train_img_wh[1],
        scale=0.2,
    )


@torch.inference_mode()
def render_image_from_camera(camera: viser.CameraState) -> None:
    W = 512
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
    viewdirs = (viewdirs[..., None, :] * torch.from_numpy(c2w[:3, :3]).to(device)).sum(
        dim=-1
    )
    origins = torch.broadcast_to(
        torch.tensor(c2w[:3, -1], dtype=torch.float32, device=device),
        viewdirs.shape,
    )
    rays = Rays(origins, viewdirs)

    rgb, acc, depth, _ = render_image(
        radiance_field,
        occupancy_grid,
        rays,
        scene_aabb=scene_aabb,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=1.0,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
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
