#!/usr/bin/env python3

"""Hello world example for viser.

The simplest possible viser program - creates a server and adds a red sphere.
"""

import viser

server = viser.ViserServer()
server.scene.add_icosphere(
    name="hello_sphere",
    radius=0.5,
    color=(255, 0, 0),  # Red
    position=(0.0, 0.0, 0.0),
)

print("Open your browser to http://localhost:8080")
print("Press Ctrl+C to exit")

while True:
    pass
