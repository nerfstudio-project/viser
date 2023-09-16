import asyncio
import multiprocessing as mp
import threading
import time
from multiprocessing.managers import DictProxy
from typing import Callable, Optional

import requests


class _ViserTunnel:
    """Tunneling utility for internal use."""

    def __init__(self, local_port: int) -> None:
        self._local_port = local_port
        self._process: Optional[mp.Process] = None

        manager = mp.Manager()
        self._shared_state = manager.dict()
        self._shared_state["status"] = "ready"
        self._shared_state["url"] = None

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Establish the tunnel connection.

        Returns URL if tunnel succeeds, otherwise None."""
        assert self._process is None

        self._shared_state["status"] = "connecting"

        self._process = mp.Process(
            target=_connect_job,
            daemon=True,
            args=(self._local_port, self._shared_state),
        )
        self._process.start()

        def wait_job() -> None:
            while self._shared_state["status"] == "connecting":
                time.sleep(0.1)
            callback()

        threading.Thread(target=wait_job).start()

    def get_url(self) -> Optional[str]:
        """Get tunnel URL. None if not connected (or connection failed)."""
        return self._shared_state["url"]

    def close(self) -> None:
        """Close the tunnel."""
        if self._process is not None:
            self._process.kill()
            self._process.join()


def _connect_job(local_port: int, shared_state: DictProxy) -> None:
    event_loop = asyncio.new_event_loop()
    assert event_loop is not None
    asyncio.set_event_loop(event_loop)
    try:
        event_loop.run_until_complete(_make_tunnel(local_port, shared_state))
    except KeyboardInterrupt:
        tasks = asyncio.all_tasks(event_loop)
        for task in tasks:
            task.cancel()
        event_loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        event_loop.close()


async def _make_tunnel(local_port: int, shared_state: DictProxy) -> None:
    share_domain = "share.viser.studio"

    try:
        response = requests.request(
            "GET",
            url=f"https://{share_domain}/?request_forward",
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            shared_state["status"] = "failed"
            return
    except requests.exceptions.ConnectionError:
        shared_state["status"] = "failed"
        return
    except Exception as e:
        shared_state["status"] = "failed"
        raise e

    res = response.json()
    shared_state["url"] = res["url"]
    shared_state["status"] = "connected"

    def make_connection_task():
        return asyncio.create_task(
            connect(
                "127.0.0.1",
                local_port,
                share_domain,
                res["port"],
            )
        )

    connection_tasks = [make_connection_task() for _ in range(res["max_conn_count"])]
    await asyncio.gather(*connection_tasks)


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    while True:
        data = await r.read(4096)
        if len(data) == 0:
            # Done!
            break
        w.write(data)
        await w.drain()


async def connect(
    local_host: str,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> None:
    """Establish a connection to the tunnel server."""

    while True:
        local_w = None
        remote_w = None
        try:
            local_r, local_w = await asyncio.open_connection(local_host, local_port)
            remote_r, remote_w = await asyncio.open_connection(remote_host, remote_port)
            await asyncio.wait(
                [
                    asyncio.create_task(pipe(local_r, remote_w)),
                    asyncio.create_task(pipe(remote_r, local_w)),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:
            pass
        finally:
            if local_w is not None:
                local_w.close()
            if remote_w is not None:
                remote_w.close()


if __name__ == "__main__":
    tunnel = _ViserTunnel(8080)
    tunnel.on_connect(lambda: None)

    time.sleep(2.0)
    print("Trying to close")
    tunnel.close()
    print("Done trying to close")
    time.sleep(10.0)
    print("Exiting")
