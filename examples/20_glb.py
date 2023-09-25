import time
from pathlib import Path

import numpy as onp
import tyro

import viser
import viser.transforms as tf


def main(
    glb_path: Path,
    scale: float,
) -> None:
    server = viser.ViserServer()
    server.add_glb(
        "glb",
        glb_path.read_bytes(),
        scale=scale,
        wxyz=tf.SO3.from_x_radians(onp.pi / 2.0).wxyz,
    )
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)
