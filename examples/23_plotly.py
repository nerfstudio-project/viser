"""Plotly.

Examples of visualizing plotly plots in Viser."""

import time

import numpy as onp
import plotly.express as px
import plotly.graph_objects as go
import viser
from PIL import Image

server = viser.ViserServer()

# We want to showcase multiple types of plots, so we'll have a dropdown to switch between them.
# Note that this isn't an exhaustive list of plot types!
# We should be able to support all plots in https://plotly.com/python/.
plot_dropdown = server.add_gui_dropdown(
    "Plot",
    options=["Line", "Image", "3D Scatter"],
    initial_value="Line",
)


# Plot type 1: Line plot.
def create_sinusoidal_wave(t: float) -> go.Figure:
    """Create a sinusoidal wave plot, starting at time t."""
    x_data = onp.linspace(t, t + 6 * onp.pi, 50)
    y_data = onp.sin(x_data) * 10

    fig = px.line(
        x=list(x_data),
        y=list(y_data),
        labels={"x": "x", "y": "sin(x)"},
        title="Sinusoidal Wave",
    )

    # this sets the margins to be tight around the title.
    fig.layout.title.automargin = True  # type: ignore
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
    )  # Reduce plot margins.

    return fig


line_plot_time = 0.0
line_plot = server.add_gui_plotly(
    figure=create_sinusoidal_wave(line_plot_time),
    visible=True,
)
line_plot.font = "Inter"  # You can also update the font family of the plot.


# Plot type 2: Image plot.
fig = px.imshow(Image.open("examples/assets/Cal_logo.png"))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)
image_plot = server.add_gui_plotly(
    figure=fig,
    aspect_ratio=1.0,
    visible=False,
)


# Plot type 3: 3D Scatter plot.
fig = px.scatter_3d(
    px.data.iris(),
    x="sepal_length",
    y="sepal_width",
    z="petal_width",
    color="species",
)
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)
scatter_plot = server.add_gui_plotly(
    figure=fig,
    aspect_ratio=1.0,
    visible=False,
)


@plot_dropdown.on_update
def _(_) -> None:
    """Callback for when the plot dropdown changes."""
    if plot_dropdown.value == "Image":
        image_plot.visible = True
        line_plot.visible = False
        scatter_plot.visible = False
    elif plot_dropdown.value == "Line":
        image_plot.visible = False
        line_plot.visible = True
        scatter_plot.visible = False
    elif plot_dropdown.value == "3D Scatter":
        image_plot.visible = False
        line_plot.visible = False
        scatter_plot.visible = True


while True:
    # Update the line plot.
    line_plot_time += 0.1
    line_plot.figure = create_sinusoidal_wave(line_plot_time)

    time.sleep(0.01)
