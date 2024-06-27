from __future__ import annotations

import asyncio
import dataclasses
import threading
from asyncio.events import AbstractEventLoop
from typing import AsyncGenerator, Dict, List, Sequence

from ._messages import Message


@dataclasses.dataclass
class AsyncMessageBuffer:
    """Async iterable for keeping a persistent buffer of messages.

    Uses heuristics on message names to automatically cull out redundant messages."""

    event_loop: AbstractEventLoop
    persistent_messages: bool
    message_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    flush_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    message_counter: int = 0
    message_from_id: Dict[int, Message] = dataclasses.field(default_factory=dict)
    id_from_redundancy_key: Dict[str, int] = dataclasses.field(default_factory=dict)

    buffer_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    """Lock to prevent race conditions when pushing messages from different threads."""

    max_window_size: int = 1024
    window_duration_sec: float = 1.0 / 60.0
    done: bool = False

    def push(self, message: Message) -> None:
        """Push a new message to our buffer, and remove old redundant ones."""

        assert isinstance(message, Message)

        # Add message to buffer.
        redundancy_key = message.redundancy_key()
        with self.buffer_lock:
            new_message_id = self.message_counter
            self.message_from_id[new_message_id] = message
            self.message_counter += 1

            # If an existing message with the same key already exists in our buffer, we
            # don't need the old one anymore. :-)
            if (
                redundancy_key is not None
                and redundancy_key in self.id_from_redundancy_key
            ):
                old_message_id = self.id_from_redundancy_key.pop(redundancy_key)
                self.message_from_id.pop(old_message_id)
            self.id_from_redundancy_key[redundancy_key] = new_message_id

        # Pulse message event to notify consumers that a new message is available.
        self.event_loop.call_soon_threadsafe(self.message_event.set)

    def flush(self) -> None:
        """Flush the message buffer; signals to yield a message window immediately."""
        self.event_loop.call_soon_threadsafe(self.flush_event.set)

    def set_done(self) -> None:
        """Set the done flag. Kills the generator."""
        self.done = True

        # Pulse message event to make sure we aren't waiting for a new message.
        self.event_loop.call_soon_threadsafe(self.message_event.set)

        # Pulse flush event to skip any windowing delay.
        self.event_loop.call_soon_threadsafe(self.flush_event.set)

    async def window_generator(
        self, client_id: int
    ) -> AsyncGenerator[Sequence[Message], None]:
        """Async iterator over messages. Loops infinitely, and waits when no messages
        are available."""

        last_sent_id = -1
        flush_wait = asyncio.create_task(self.flush_event.wait())
        while not self.done:
            window: List[Message] = []
            most_recent_message_id = self.message_counter - 1
            while (
                last_sent_id < most_recent_message_id
                and len(window) < self.max_window_size
            ):
                last_sent_id += 1
                if self.persistent_messages:
                    message = self.message_from_id.get(last_sent_id, None)
                else:
                    # If we're not persisting messages, remove them from the buffer.
                    with self.buffer_lock:
                        message = self.message_from_id.pop(last_sent_id, None)
                        if message is not None:
                            redundancy_key = message.redundancy_key()
                            self.id_from_redundancy_key.pop(redundancy_key, None)

                if message is not None and message.excluded_self_client != client_id:
                    window.append(message)

            if len(window) > 0:
                # Yield a window!
                yield window
            else:
                # Wait for a new message to come in.
                await self.message_event.wait()
                self.message_event.clear()

            # Add a delay if either (a) we failed to yield or (b) there's currently no messages to send.
            most_recent_message_id = self.message_counter - 1
            if len(window) == 0 or most_recent_message_id == last_sent_id:
                done, pending = await asyncio.wait(
                    [flush_wait], timeout=self.window_duration_sec
                )
                del pending
                if flush_wait in done and not self.done:
                    self.flush_event.clear()
                    flush_wait = asyncio.create_task(self.flush_event.wait())
