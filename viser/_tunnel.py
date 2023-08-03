"""Helper for creating shareable viser links using `bore`.

https://github.com/ekzhang/bore
"""

import atexit
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

BORE_SERVER = "chalupa.eecs.berkeley.edu"


def start_tunnel(port: int) -> None:
    env_dir = Path(__file__).absolute().parent / ".rustenv"
    bore_path = env_dir / "rust" / "bin" / "bore"

    if not env_dir.exists():
        print("[viser] Tunneling: setting up Rust environment")
        subprocess.run(args=[sys.executable, "-m", "rustenv", str(env_dir)])

    if not bore_path.exists():
        print("[viser] Tunneling: installing bore")
        subprocess.run(args=[str(env_dir / "bin" / "cargo"), "install", "bore-cli"])

    process = subprocess.Popen(
        [f"{bore_path}", "local", str(port), "--to", BORE_SERVER],
        stdout=subprocess.PIPE,
    )

    # Handle normal exists.
    @atexit.register
    def _():
        process.terminate()
        process.wait()

    # Handle SIGTERM.
    script_pid = os.getpid()

    def handle_termination_signal(signum, frame):
        process.kill()
        os.kill(script_pid, signum)

    signal.signal(signal.SIGTERM, handle_termination_signal)

    def _bore_watcher() -> None:
        while True:
            assert process.stdout is not None
            line = process.stdout.readline().decode().strip()
            if "Error:" in line:
                print(line)
                break
            if "listening at " in line:
                print("[viser] Tunnel created at", line.partition("listening at ")[2])

    threading.Thread(target=_bore_watcher).start()

    return None
