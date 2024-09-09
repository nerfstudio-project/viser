"""Lights

Visualize a mesh under different lighting conditions. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

import time
from pathlib import Path

import numpy as onp
import trimesh

import viser
import viser.transforms as tf


def main() -> None:
    # Load mesh.
    mesh = trimesh.load_mesh(str(Path(__file__).parent / "assets/dragon.obj"))
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.apply_scale(0.05)
    vertices = mesh.vertices
    faces = mesh.faces
    print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

    # Start Viser server with mesh.
    server = viser.ViserServer()

    server.scene.add_mesh_simple(
        name="/simple",
        vertices=vertices,
        faces=faces,
        wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_mesh_trimesh(
        name="/trimesh",
        mesh=mesh,
        wxyz=tf.SO3.from_x_radians(onp.pi / 2).wxyz,
        position=(0.0, 5.0, 0.0),
    )

    # adding controls to custom lights in the scene
    server.scene.add_transform_controls("/light_control")
    server.scene.add_light_directional(
        name="/light_control/directionallight", color=0xDEADBE
    )
    server.scene.add_transform_controls("/light_control2")
    server.scene.add_light_spot(
        name="/light_control2/spotlight",
        color=0xC0FFEE,
        distance=5,
        angle=onp.pi / 2.5,
        intensity=3,
    )

    # Create default light toggle.
    gui_default_lights = server.gui.add_checkbox("Default lights", initial_value=True)
    gui_default_lights.on_update(
        lambda _: server.scene.enable_default_lights(gui_default_lights.value)
    )

    # Create GUI elements for controlling environment map.
    with server.gui.add_folder("Environment map"):
        gui_env_preset = server.gui.add_dropdown(
            "Preset",
            (
                "None",
                "apartment",
                "city",
                "dawn",
                "forest",
                "lobby",
                "night",
                "park",
                "studio",
                "sunset",
                "warehouse",
            ),
            initial_value="city",
        )
        gui_background = server.gui.add_checkbox("Background", False)
        gui_bg_blurriness = server.gui.add_slider(
            "Bg Blurriness",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )
        gui_bg_intensity = server.gui.add_slider(
            "Bg Intensity",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=1.0,
        )
        gui_env_intensity = server.gui.add_slider(
            "Env Intensity",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=1.0,
        )

    def update_environment_map(_) -> None:
        server.scene.set_environment_map(
            gui_env_preset.value if gui_env_preset.value != "None" else None,
            background=gui_background.value,
            background_blurriness=gui_bg_blurriness.value,
            background_intensity=gui_bg_intensity.value,
            environment_intensity=gui_env_intensity.value,
        )

    gui_env_preset.on_update(update_environment_map)
    gui_background.on_update(update_environment_map)
    gui_bg_blurriness.on_update(update_environment_map)
    gui_bg_intensity.on_update(update_environment_map)
    gui_env_intensity.on_update(update_environment_map)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
