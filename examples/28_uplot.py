"""uPlot

Examples of visualizing uPlot plots in Viser."""

import time

import numpy as np

import viser


def y0(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x) + 0.1 * np.random.normal(size=x.shape)


def y1(x: np.ndarray) -> np.ndarray:
    return -y0(x) + 1.0


def main() -> None:
    server = viser.ViserServer()

    time_step = 1.0 / 60.0
    x_data = time_step * np.arange(100, dtype=np.float64)
    y0_data = y0(x_data)
    y1_data = y1(x_data)

    # Data for uPlot: tuple of arrays where first is x-data, rest are y-data
    data = (x_data, y0_data, y1_data)

    print("data shapes:", [arr.shape for arr in data])

    uplot_handle = server.gui.add_uplot(
        data=data,
        series=(
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
        ),
        scales={
            "x": {
                "time": False,
                "auto": True,
            },
            "y": {"range": [-1.5, 2.5]},
        },
        legend={"show": True},
        aspect=1.0,
    )

    while True:
        # Update the line plot.
        x_data = data[0] + time_step
        y0_data = y0(x_data)
        y1_data = y1(x_data)

        # Update the data
        data = (x_data, y0_data, y1_data)
        uplot_handle.data = data
        time.sleep(time_step)


if __name__ == "__main__":
    main()
