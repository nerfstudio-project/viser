#!/usr/bin/env python


import asyncio
import multiprocessing
import threading

import websockets.connection
import websockets.exceptions
import websockets.server

from ._async_message_buffer import AsyncMessageBuffer
from ._messages import Message


class ViserServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        # Create websocket server process.
        self.message_queue = multiprocessing.Queue(maxsize=1024)
        multiprocessing.Process(
            target=self._start_background_loop,
            args=(host, port, self.message_queue),
            daemon=True,
        ).start()

    def queue(self, *messages: Message) -> None:
        """Queue a message, which will be sent to all clients."""
        for message in messages:
            self.message_queue.put_nowait(message)

    def _start_background_loop(
        self,
        host: str,
        port: int,
        message_queue: multiprocessing.Queue,
    ) -> None:
        connection_count = 0
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        message_buffer = AsyncMessageBuffer(event_loop)

        def message_transfer() -> None:
            """Message transfer loop. Pulls messages from the main process and pushes them into our buffer."""
            while True:
                message_buffer.push(message_queue.get())

        async def serve(websocket: websockets.server.WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""
            nonlocal connection_count
            connection_id = connection_count
            connection_count += 1

            print(
                f"Connection opened ({connection_id}),"
                f" {len(message_buffer.message_from_id)} buffered messages"
            )

            # Infinite loop over messages from the message buffer.
            try:
                async for message in message_buffer:
                    await websocket.send(message)
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError,
            ):
                print(f"Connection closed ({connection_id})")

        # Start message transfer thread.
        threading.Thread(target=message_transfer).start()

        # Run server.
        event_loop.run_until_complete(websockets.server.serve(serve, host, port))
        event_loop.run_forever()
