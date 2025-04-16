"""Images

Example for sending images to the viewer.

We can send backgrond images to display behind the viewer (useful for visualizing
NeRFs), or images to render as 3D textures.
"""

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

import viser


def main() -> None:
    server = viser.ViserServer()

    # Add a background image.
    server.scene.set_background_image(
        iio.imread(Path(__file__).parent / "assets/Cal_logo.png"),
        format="png",
    )

    # Add main image.
    server.scene.add_image(
        "/img",
        iio.imread(Path(__file__).parent / "assets/Cal_logo.png"),
        4.0,
        4.0,
        format="png",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(2.0, 2.0, 0.0),
    )
    while True:
        server.scene.add_image(
            "/noise",
            np.random.randint(0, 256, size=(400, 400, 3), dtype=np.uint8),
            4.0,
            4.0,
            format="jpeg",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(2.0, 2.0, -1e-2),
        )
        time.sleep(0.2)


if __name__ == "__main__":
    main()
