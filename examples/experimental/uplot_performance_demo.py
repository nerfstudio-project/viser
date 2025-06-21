"""Uplot

Examples of visualizing uplot plots in Viser."""

from __future__ import annotations

import time

import numpy as np

import viser
import viser.uplot


def y(x: np.ndarray, offset: int) -> np.ndarray:
    return 0.5 * np.sin(2 * np.pi * x + offset) + 0.5 * np.random.normal(size=x.shape)


def main() -> None:
    server = viser.ViserServer()

    num_plots = 5
    num_trajectories = 5
    num_points = 100
    time_step = 1.0 / 60.0

    x_data = time_step * np.arange(num_points, dtype=np.float64)
    uplot_handles: list[viser.GuiUplotHandle] = []
    for _ in range(num_plots):
        uplot_handles.append(
            server.gui.add_uplot(
                data=(x_data, *[y(x_data, i) for i in range(num_trajectories)]),
                series=(
                    viser.uplot.Series(label="time"),
                    *[
                        viser.uplot.Series(
                            label=f"y{i + 1}",
                            stroke=["red", "green", "blue", "orange", "purple"][i % 5],
                            width=2,
                        )
                        for i in range(num_trajectories)
                    ],
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(range=(-2, 2)),
                },
                legend=viser.uplot.Legend(show=True),
                aspect=1.0,
            )
        )

    while True:
        # Update the line plots.
        x_data = x_data + time_step
        for plot_index in range(num_plots):
            uplot_handles[plot_index].data = (
                x_data,
                *[y(x_data, i) for i in range(num_trajectories)],
            )
        time.sleep(time_step)


if __name__ == "__main__":
    main()
