"""Plotly Demonstration
"""

import time
from pathlib import Path

import viser

server = viser.ViserServer()

title = time.time()

data = [
    {
        "x": [1, 2],
        "y": [10, 20],
        "type": "scatter",
    },
    {
        "x": [1, 2],
        "y": [10, 20],
        "type": "bar",
    },
]

markdown_blurb = server.add_gui_plotly(title=title, data=data)

number = server.add_gui_number("Number", initial_value=5)

while True:
    # data['x'] = data['x'] + [data['x'][-1] + 1]
    # data['y'] = data['y'] + [data['y'][-1] * -1]
    data[0]["x"] = data[0]["x"] + [data[0]["x"][-1] + 1]
    data[0]["y"] = data[0]["y"] + [data[0]["y"][-1] * -1]
    markdown_blurb.data = data
    # markdown_blurb.title = str(time.time())
    number.value = time.time()

    time.sleep(1)
