"""Websocket-based communication infrastructure.

We implement abstractions for:
- Launching a websocket+HTTP server on a shared port.
- Registering callbacks for connection events and incoming messages.
- Asynchronous message sending, both broadcasted and to individual clients.
- Defining dataclass-based message types.
- Translating Python message types to TypeScript interfaces.

These are what `viser` runs on under-the-hood, and generally won't be useful unless
you're building your own front-end from scratch.
"""

from ._core import ClientConnection as ClientConnection
from ._core import ClientId as ClientId
from ._core import MessageHandler as MessageHandler
from ._core import Server as Server
from ._messages import Message as Message
from ._typescript_interface_gen import (
    generate_typescript_interfaces as generate_typescript_interfaces,
)
