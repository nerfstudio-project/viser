import asyncio
import dataclasses
import time
from asyncio.events import AbstractEventLoop
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Dict,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

from typing_extensions import Literal, TypeGuard

from ._messages import Message


@dataclasses.dataclass
class AsyncMessageBuffer:
    """Async iterable for keeping a persistent buffer of messages.

    Uses heuristics on message names to automatically cull out redundant messages."""

    event_loop: AbstractEventLoop
    message_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    _flush_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

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

    def flush(self) -> None:
        """Immediately send any buffered messages."""
        self._flush_event.set()

    async def window_generator(
        self, client_id: int
    ) -> AsyncGenerator[Sequence[Message], None]:
        """Async iterator over messages. Loops infinitely, and waits when no messages
        are available."""
        # Wait for a first message to arrive.
        if len(self.message_from_id) == 0:
            await self.message_event.wait()

        window = MessageWindow(client_id=client_id)
        last_sent_id = -1
        while True:
            # Wait until there are new messages available.
            most_recent_message_id = self.message_counter - 1
            while last_sent_id >= most_recent_message_id:
                next_message = self.message_event.wait()
                send_window = False
                try:
                    flush_wait = self._flush_event.wait()
                    done, pending = await asyncio.wait(
                        [flush_wait, next_message],
                        timeout=window.max_time_until_ready(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    del pending
                    if flush_wait in done:
                        send_window = True
                        self._flush_event.clear()
                    most_recent_message_id = self.message_counter - 1
                except asyncio.TimeoutError:
                    send_window = True

                if send_window:
                    out = window.get_window_to_send()
                    if out is not None:
                        yield out

            # Try to yield the next message ID. Note that messages can be culled before
            # they're sent.
            last_sent_id += 1
            message = self.message_from_id.get(last_sent_id, None)
            if message is not None:
                window.append_to_window(message)
                self.event_loop.call_soon_threadsafe(self.message_event.clear)

                # Sleep to yield.
                await asyncio.sleep(1e-8)

            out = window.get_window_to_send()
            if out is not None:
                yield out


DONE_SENTINEL = "done_sentinel"
DoneSentinel = Literal["done_sentinel"]


def is_done_sentinel(x: Any) -> TypeGuard[DoneSentinel]:
    return x == DONE_SENTINEL


@dataclasses.dataclass
class MessageWindow:
    """Helper for building windows of messages to send to clients."""

    client_id: int
    """Client that this window will be sent to. Used for ignoring certain messages."""

    window_duration_sec: float = 1.0 / 60.0
    window_max_length: int = 256

    done: bool = False

    _window_start_time: float = -1
    _window: Dict[str, Message] = dataclasses.field(default_factory=dict)
    """We use a redundancy key -> message dictionary to track our window. This helps us
    eliminate redundant messages."""

    def append_to_window(self, message: Union[Message, DoneSentinel]) -> None:
        """Append a message to our window."""
        if isinstance(message, Message):
            if message.excluded_self_client == self.client_id:
                return
            if len(self._window) == 0:
                self._window_start_time = time.time()
            self._window[message.redundancy_key()] = message
        else:
            assert is_done_sentinel(message)
            self.done = True

    async def wait_and_append_to_window(
        self,
        message: Awaitable[Union[Message, DoneSentinel]],
        flush_event: asyncio.Event,
    ) -> bool:
        """Async version of `append_to_window()`. Returns `True` if successful, `False`
        if timed out."""
        if len(self._window) == 0:
            self.append_to_window(await message)
            return True

        message = asyncio.shield(message)
        flush_wait = asyncio.shield(flush_event.wait())
        (done, pending) = await asyncio.wait(
            [message, flush_wait],
            timeout=self.max_time_until_ready(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        del pending
        if flush_wait in done:
            flush_event.clear()
        flush_wait.cancel()
        if message in cast(Set[Any], done):  # Cast to prevent type narrowing.
            self.append_to_window(await message)
            return True
        return False

    def max_time_until_ready(self) -> Optional[float]:
        """Returns the maximum amount of time, in seconds, until we're ready to send the
        current window. If the window is empty, returns `None`."""
        if len(self._window) == 0:
            return None
        elapsed = time.time() - self._window_start_time
        return max(0.0, self.window_duration_sec - elapsed)

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
        if not ready:
            return None
        out = tuple(self._window.values())
        self._window.clear()
        return out
