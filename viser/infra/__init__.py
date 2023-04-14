"""Application-agnostic communication for viser.

We specify interfaces for:
    - Launching a websocket+HTTP server on a shared port.
    - Registering callbacks for new connections and incoming messages.
    - Defining dataclass-based message types.
    - Translating Python message types to TypeScript interfaces.
"""

from ._core import ClientConnection as ClientConnection
from ._core import ClientId as ClientId
from ._core import MessageHandler as MessageHandler
from ._core import Server as Server
from ._messages import Message as Message
from ._typescript_interface_gen import (
    generate_typescript_interfaces as generate_typescript_interfaces,
)
