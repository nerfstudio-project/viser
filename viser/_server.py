#!/usr/bin/env python


import asyncio
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
        wait_for_connection: bool = False,
    ):
        # Create websocket server thread.
        event_loop = asyncio.get_event_loop()
        threading.Thread(
            target=self._start_background_loop,
            args=(host, port, event_loop),
            daemon=True,
        ).start()
        self.event_loop = event_loop
        self.message_buffer = AsyncMessageBuffer()
        self.message_event = asyncio.Event()

        # Wait for client to connect.
        self.connection_sem = threading.Semaphore(value=0)
        if wait_for_connection:
            self.connection_sem.acquire()

    def queue(self, *messages: Message) -> None:
        """Queue a message, which will be sent to all clients."""
        for message in messages:
            self.message_buffer.push(message)
        self.message_buffer.notify(self.event_loop)

    def _start_background_loop(
        self, host: str, port: int, loop: asyncio.AbstractEventLoop
    ) -> None:
        connection_count = 0

        async def serve(websocket: websockets.server.WebSocketServerProtocol) -> None:
            """Server loop, run once per connection."""
            nonlocal connection_count
            connection_id = connection_count
            connection_count += 1

            print(
                f"Connection opened ({connection_id}),"
                f" {len(self.message_buffer.message_from_id)} buffered messages"
            )

            # Infinite loop over messages from the message buffer.
            try:
                async for message in self.message_buffer:
                    await websocket.send(message)
            except websockets.exceptions.ConnectionClosedOK:
                print(f"Connection closed ({connection_id})")

        asyncio.set_event_loop(loop)
        start_server = websockets.server.serve(
            serve,
            host,
            port=port,
        )
        loop.run_until_complete(start_server)
        loop.run_forever()
