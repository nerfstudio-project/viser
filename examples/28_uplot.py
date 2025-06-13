"""Uplot

Examples of visualizing uplot plots in Viser."""

import time

import numpy as np

import viser


def y0(x: float | np.ndarray) -> float | np.ndarray:
    length = 1 if isinstance(x, float) else len(x)
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn(length)


def y1(x: float | np.ndarray) -> float | np.ndarray:
    return -y0(x) + 1.0


def main() -> None:
    server = viser.ViserServer(port=8100)

    options = {
        "scales": {
            "x": {
                "time": False,
                "auto": True,
            },
            "y": {"range": [-1.5, 2.5]},
        },
        "axes": [{}],
        "series": [
            {"label": "time"},
            {
                "label": "y0",
                "stroke": "blue",
                "width": 2,
            },
            {
                "label": "y1",
                "stroke": "red",
                "width": 2,
            },
        ],
        "legend": {"show": True},
    }

    time_step = 1.0 / 60.0
    x_data = time_step * np.arange(100)
    y0_data = y0(x_data)
    y1_data = y1(x_data)
    aligned_data = np.vstack((x_data, y0_data, y1_data))

    print("aligned_data.shape", aligned_data.shape)
    print("aligned_data", aligned_data)
    uplot_handle = server.gui.add_uplot(
        aligned_data=aligned_data,
        options=options,
        aspect=1.0,
    )

    while True:
        # Update the line plot.
        x_data = aligned_data[0, :] + time_step
        y0_data = np.concatenate((aligned_data[1, 1:], y0(x_data[-1])), axis=0)
        y1_data = np.concatenate((aligned_data[2, 1:], y1(x_data[-1])), axis=0)
        aligned_data = np.vstack((x_data, y0_data, y1_data))
        uplot_handle.aligned_data = aligned_data
        time.sleep(time_step)


if __name__ == "__main__":
    main()
