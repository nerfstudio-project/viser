"""uPlot

Examples of visualizing uPlot plots in Viser."""

import time

import numpy as np

import viser
import viser.uplot


def y0(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x) + 0.1 * np.random.normal(size=x.shape)


def y1(x: np.ndarray) -> np.ndarray:
    return -y0(x) + 1.0


def main() -> None:
    server = viser.ViserServer()

    time_step = 1.0 / 60.0

    # Data for uPlot: tuple of arrays where first is x-data, rest are y-data
    x_data = time_step * np.arange(100)
    data = (x_data, y0(x_data), y1(x_data))

    print("data shapes:", [arr.shape for arr in data])

    uplot_handle = server.gui.add_uplot(
        data=data,
        series=(
            viser.uplot.Series(label="time"),
            viser.uplot.Series(
                label="y0",
                stroke="blue",
                width=2,
            ),
            viser.uplot.Series(
                label="y1",
                stroke="red",
                width=2,
            ),
        ),
        scales={
            "x": viser.uplot.Scale(
                time=False,
                auto=True,
            ),
            "y": viser.uplot.Scale(range=(-1.5, 2.5)),
        },
        legend=viser.uplot.Legend(show=True),
        aspect=1.0,
    )

    while True:
        # Update the line plot.
        x_data = x_data + time_step
        uplot_handle.data = (x_data, y0(x_data), y1(x_data))
        time.sleep(time_step)


if __name__ == "__main__":
    main()
