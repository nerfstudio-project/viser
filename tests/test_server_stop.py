import socket
import time

import viser
import viser._client_autobuild


def test_server_port_is_freed():
    # Mock the client autobuild to avoid building the client.
    viser._client_autobuild.ensure_client_is_built = lambda: None

    server = viser.ViserServer()
    original_port = server.get_port()

    # Assert that the port is not free.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", original_port))
    assert result == 0
    sock.close()
    server.stop()

    time.sleep(0.05)

    # Assert that the port is now free.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", original_port))
    assert result != 0
