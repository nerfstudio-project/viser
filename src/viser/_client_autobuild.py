import os
import subprocess
import sys
from pathlib import Path

import rich

client_dir = Path(__file__).absolute().parent / "client"
build_dir = client_dir / "build"


def _check_viser_yarn_running() -> bool:
    """Returns True if the viewer client has been launched via `yarn start`."""
    import psutil

    for process in psutil.process_iter():
        try:
            if Path(process.cwd()).as_posix().endswith("viser/client") and any(
                [part.endswith("yarn") for part in process.cmdline()]
                + [part.endswith("yarn.js") for part in process.cmdline()]
            ):
                return True
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def ensure_client_is_built() -> None:
    """Ensure that the client is built or already running."""

    if not (client_dir / "src").exists():
        # Can't build client.
        assert (build_dir / "index.html").exists(), (
            "Something went wrong! At least one of the client source or build"
            " directories should be present."
        )
        return

    # Do we need to re-trigger a build?
    build = False
    if _check_viser_yarn_running():
        # Don't run `yarn build` if `yarn start` is already running.
        rich.print(
            "[bold](viser)[/bold] The Viser viewer looks like it has been launched via"
            " `yarn start`. Skipping build check..."
        )
        build = False
    elif not (build_dir / "index.html").exists():
        rich.print("[bold](viser)[/bold] No client build found. Building now...")
        build = True
    elif _modified_time_recursive(client_dir / "src") > _modified_time_recursive(
        build_dir
    ):
        rich.print(
            "[bold](viser)[/bold] Client build looks out of date. Building now..."
        )
        build = True

    # Install nodejs and build if necessary. We assume bash is installed.
    if build:
        node_bin_dir = _install_sandboxed_node()
        npx_path = node_bin_dir / "npx"

        subprocess_env = os.environ.copy()
        subprocess_env["NODE_VIRTUAL_ENV"] = str(node_bin_dir.parent)
        subprocess_env["PATH"] = (
            str(node_bin_dir)
            + (";" if sys.platform == "win32" else ":")
            + subprocess_env["PATH"]
        )
        subprocess.run(
            args=f"{npx_path} --yes yarn install",
            env=subprocess_env,
            cwd=client_dir,
            shell=True,
            check=False,
        )
        subprocess.run(
            args=f"{npx_path} --yes yarn run build",
            env=subprocess_env,
            cwd=client_dir,
            shell=True,
            check=False,
        )


def _install_sandboxed_node() -> Path:
    """Install a sandboxed copy of nodejs using nodeenv, and return a path to the
    environment's bin directory (`.nodeenv/bin` or `.nodeenv/Scripts`).

    On Windows, the `.nodeenv/bin` does not exist. Instead, executables are
    installed to `.nodeenv/Scripts`."""

    def get_node_bin_dir() -> Path:
        env_dir = client_dir / ".nodeenv"
        node_bin_dir = env_dir / "bin"
        if not node_bin_dir.exists():
            node_bin_dir = env_dir / "Scripts"
        return node_bin_dir

    node_bin_dir = get_node_bin_dir()
    if (node_bin_dir / "npx").exists():
        rich.print("[bold](viser)[/bold] nodejs is set up!")
        return node_bin_dir

    env_dir = client_dir / ".nodeenv"
    subprocess.run(
        [sys.executable, "-m", "nodeenv", "--node=20.4.0", env_dir], check=False
    )

    node_bin_dir = get_node_bin_dir()
    assert (node_bin_dir / "npx").exists()
    return node_bin_dir


def _modified_time_recursive(dir: Path) -> float:
    """Recursively get the last time a file was modified in a directory."""
    return max([f.stat().st_mtime for f in dir.glob("**/*")])
