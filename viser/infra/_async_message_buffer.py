import asyncio
import dataclasses
from asyncio.events import AbstractEventLoop
from typing import AsyncGenerator, Dict

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

    async def __aiter__(self) -> AsyncGenerator[Message, None]:
        """Async iterator over messages. Loops infinitely, and waits when no messages
        are available."""
        # Wait for a first message to arrive.
        if len(self.message_from_id) == 0:
            await self.message_event.wait()

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
            if message is not None:
                yield message
                # TODO: it's likely OK for now, but feels sketchy to be sharing the same
                # message event across all consumers.
                self.event_loop.call_soon_threadsafe(self.message_event.clear)

                # Small sleep to invoke a yield.
                await asyncio.sleep(1e-8)
