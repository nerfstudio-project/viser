import random
import time

import viser

server = viser.ViserServer()

while True:
    server.queue(
        viser.ResetSceneMessage(),
        viser.FrameMessage(
            "/tree",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
        viser.FrameMessage(
            "/tree/branch",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
        viser.FrameMessage(
            "/tree/branch/leaf",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        ),
    )
    time.sleep(5.0)
    server.queue(viser.RemoveSceneNodeMessage("/tree/branch"))
    time.sleep(0.5)
