"""Uplot

Examples of visualizing uplot plots in Viser."""

import time

import numpy as np

import viser


def y0(x: float | np.ndarray) -> float | np.ndarray:
    length = 1 if isinstance(x, float) else len(x)
    return 0.5 * np.sin(2 * np.pi * x) + 0.5 * np.random.randn(length)


def main() -> None:
    server = viser.ViserServer(port=8100)

    num_plots = 10
    num_trajectories = 5
    num_points = 100
    time_step = 1.0 / 60.0

    options = {
        "scales": {
            "x": {
                "time": False,
                "auto": True,
            },
            "y": {"range": [-2, 2]},
        },
        "axes": [{}],
        "series": [
            {"label": "time"},
            *[
                {
                    "label": f"y{i + 1}",
                    "stroke": ["red", "green", "blue", "orange", "purple"][i % 5],
                    "width": 2,
                }
                for i in range(num_trajectories)
            ],
        ],
        "legend": {"show": True},
    }

    x_data = time_step * np.arange(num_points)
    y_data = np.vstack([y0(x_data) for _ in range(num_trajectories)])
    print("y_data.shape", y_data.shape)
    print("x_data.shape", x_data.shape)
    print("x_data", x_data)
    aligned_data = np.concatenate((x_data[None, :], y_data), axis=0)
    print("aligned_data.shape", aligned_data.shape)

    uplot_handles = []
    for _ in range(num_plots):
        uplot_handles.append(
            server.gui.add_uplot(
                aligned_data=aligned_data,
                options=options,
                aspect=1.0,
            )
        )

    while True:
        # Update the line plots.
        x_data = aligned_data[0, :] + time_step
        for plot_index in range(num_plots):
            new_y = y0(x_data[-1] * np.ones(num_trajectories))[:, None]
            y_data = np.concatenate((aligned_data[1:, 1:], new_y), axis=1)
            aligned_data = np.concatenate((x_data[None, :], y_data), axis=0)
            uplot_handles[plot_index].aligned_data = aligned_data
        time.sleep(time_step)


if __name__ == "__main__":
    main()
