import random
import time

import viser

server = viser.ViserServer()

while True:
    server.queue(
        viser.ResetSceneMessage(),
        viser.FrameMessage(
            "/tree",
            xyzw=(0.0, 0.0, 0.0, 1.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
        viser.FrameMessage(
            "/tree/branch",
            xyzw=(0.0, 0.0, 0.0, 1.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
        viser.FrameMessage(
            "/tree/branch/leaf",
            xyzw=(0.0, 0.0, 0.0, 1.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
    )
    time.sleep(0.5)
    server.queue(viser.RemoveSceneNodeMessage("/tree/branch"))
    time.sleep(0.5)
