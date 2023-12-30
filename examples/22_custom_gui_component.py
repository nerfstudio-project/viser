"""Advanced GUI - custom GUI components"""

import time
from pathlib import Path

import numpy as onp

import trimesh
import viser
import viser.transforms as tf
from viser import Icon

mesh = trimesh.load_mesh(Path(__file__).parent / "assets/dragon.obj")
assert isinstance(mesh, trimesh.Trimesh)
mesh.apply_scale(0.05)

vertices = mesh.vertices
faces = mesh.faces
print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

server = viser.ViserServer()

import_button = server.add_gui_button("Import", icon=Icon.FOLDER_OPEN)
export_button = server.add_gui_button("Export", icon=Icon.DOWNLOAD)

fps = server.add_gui_number("FPS", 24, min=1, icon=Icon.KEYFRAMES, hint="Frames per second")
duration = server.add_gui_number("Duration", 4.0, min=0.1, icon=Icon.CLOCK_HOUR_5, hint="Duration in seconds")
width = server.add_gui_number("Width", 1920, min=100, icon=Icon.ARROWS_HORIZONTAL, hint="Width in px")
height = server.add_gui_number("Height", 1080, min=100, icon=Icon.ARROWS_VERTICAL, hint="Height in px")
fov = server.add_gui_number("FOV", 75, min=1, max=179, icon=Icon.CAMERA, hint="Field of view")
smoothness = server.add_gui_slider("Smoothness", 0.5, min=0.0, max=1.0, step=0.01, hint="Trajectory smoothing")


duration = 4
cameras_slider = server.add_gui_multi_slider(
    "Timeline",
    min=0.,
    max=1.,
    step=0.01,
    initial_value=[0.0, 0.5, 1.0],
    disabled=False,
    marks=[(x, f'{x*duration:.1f}s') for x in [0., 0.5, 1.0]],
)

@duration.on_update
def _(_) -> None:
    cameras_slider.marks=[(x, f'{x*duration.value:.1f}s') for x in [0., 0.5, 1.0]],

server.add_mesh_simple(
    name="/simple",
    vertices=vertices,
    faces=faces,
    wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
    position=(0.0, 0.0, 0.0),
)
server.add_mesh_trimesh(
    name="/trimesh",
    mesh=mesh.smoothed(),
    wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
    position=(0.0, 5.0, 0.0),
)
panel = server.add_gui_camera_trajectory_panel()

while True:
    time.sleep(10.0)