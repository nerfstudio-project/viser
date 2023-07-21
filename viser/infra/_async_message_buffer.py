import asyncio
import dataclasses
import time
from asyncio.events import AbstractEventLoop
from typing import AsyncGenerator, Awaitable, Dict, List, Optional, Sequence

from ._messages import Message


@dataclasses.dataclass
class AsyncMessageBuffer:
    """Async iterable for keeping a persistent buffer of messages.

    Uses heuristics on message names to automatically cull out redundant messages."""

    event_loop: AbstractEventLoop
    message_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    message_counter: int = 0
    message_from_id: Dict[int, Message] = dataclasses.field(default_factory=dict)
    id_from_redundancy_key: Dict[str, int] = dataclasses.field(default_factory=dict)

    def push(self, message: Message) -> None:
        """Push a new message to our buffer, and remove old redundant ones."""
        # Add message to buffer.
        new_message_id = self.message_counter
        self.message_from_id[new_message_id] = message
        self.message_counter += 1

        # If an existing message with the same key already exists in our buffer, we
        # don't need the old one anymore. :-)
        redundancy_key = message.redundancy_key()
        if redundancy_key is not None and redundancy_key in self.id_from_redundancy_key:
            old_message_id = self.id_from_redundancy_key.pop(redundancy_key)
            self.message_from_id.pop(old_message_id)
        self.id_from_redundancy_key[redundancy_key] = new_message_id

        # Notify consumers that a new message is available.
        self.event_loop.call_soon_threadsafe(self.message_event.set)

    async def window_generator(
        self, ignore_client_id: int
    ) -> AsyncGenerator[Sequence[Message], None]:
        """Async iterator over messages. Loops infinitely, and waits when no messages
        are available."""
        # Wait for a first message to arrive.
        if len(self.message_from_id) == 0:
            await self.message_event.wait()

        window = MessageWindow()
        last_sent_id = -1
        while True:
            # Wait until there are new messages available.
            most_recent_message_id = self.message_counter - 1
            while last_sent_id >= most_recent_message_id:
                await self.message_event.wait()
                most_recent_message_id = self.message_counter - 1

            # Try to yield the next message ID. Note that messages can be culled before
            # they're sent.
            last_sent_id += 1
            message = self.message_from_id.get(last_sent_id, None)
            if message is not None and message.excluded_self_client != ignore_client_id:
                window.append_to_window(message)
                self.event_loop.call_soon_threadsafe(self.message_event.clear)

                # Sleep to yield.
                await asyncio.sleep(1e-8)

            out = window.get_window_to_send()
            if out is not None:
                yield out


@dataclasses.dataclass
class MessageWindow:
    """Helper for building windows of messages to send to clients."""

    window_duration_sec: float = 1.0 / 60.0
    window_max_length: int = 1024

    _window_start_time: float = -1
    _window: List[Message] = dataclasses.field(default_factory=list)

    def append_to_window(self, message: Message) -> None:
        """Append a message to our window."""
        if len(self._window) == 0:
            self._window_start_time = time.time()
        self._window.append(message)

    async def wait_and_append_to_window(self, message: Awaitable[Message]) -> None:
        """Async version of `append_to_window()`."""
        message = asyncio.shield(message)
        if len(self._window) == 0:
            self.append_to_window(await message)
        else:
            elapsed = time.time() - self._window_start_time
            (done, pending) = await asyncio.wait(
                [message], timeout=elapsed - self.window_duration_sec
            )
            del pending
            if message in done:
                self.append_to_window(await message)

    def get_window_to_send(self) -> Optional[Sequence[Message]]:
        """Returns window of messages if ready. Otherwise, returns None."""
        # Are we ready to send?
        ready = False
        if (
            len(self._window) > 0
            and time.time() - self._window_start_time >= self.window_duration_sec
        ):
            ready = True
        elif len(self._window) >= self.window_max_length:
            ready = True

        # Clear window and return if ready.
        if ready:
            out = tuple(self._window)
            self._window.clear()
            return out
        else:
            return None
