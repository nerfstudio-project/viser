"""Splines

Make a ball with some random splines.
"""

import time

import numpy as np

import viser


def main() -> None:
    server = viser.ViserServer()
    for i in range(10):
        positions = np.random.normal(size=(30, 3)) * 3.0
        server.scene.add_spline_catmull_rom(
            f"/catmull_{i}",
            positions,
            tension=0.5,
            line_width=3.0,
            color=np.random.uniform(size=3),
            segments=100,
        )

        control_points = np.random.normal(size=(30 * 2 - 2, 3)) * 3.0
        server.scene.add_spline_cubic_bezier(
            f"/cubic_bezier_{i}",
            positions,
            control_points,
            line_width=3.0,
            color=np.random.uniform(size=3),
            segments=100,
        )

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
