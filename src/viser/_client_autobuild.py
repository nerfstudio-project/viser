import subprocess
import sys
from pathlib import Path

import psutil
import rich

client_dir = Path(__file__).absolute().parent / "client"
build_dir = client_dir / "build"


def _check_viser_yarn_running() -> bool:
    """Returns True if the viewer client has been launched via `yarn start`."""
    for process in psutil.process_iter():
        try:
            if Path(process.cwd()).as_posix().endswith("viser/client") and any(
                [part.endswith("yarn") for part in process.cmdline()]
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
            "[bold](viser)[/bold] The Viser viewer looks like it has been launched via `yarn start`. Skipping build check..."
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
        env_dir = _install_sandboxed_node()
        npx_path = env_dir / "bin" / "npx"
        subprocess.run(
            args=(
                "bash -c '"
                f"source {env_dir / 'bin' / 'activate'};"
                f"{npx_path} yarn install;"
                f"{npx_path} yarn run build;"
                "'"
            ),
            cwd=client_dir,
            shell=True,
        )


def _install_sandboxed_node() -> Path:
    """Install a sandboxed copy of nodejs using nodeenv, and return a path to the
    environment root."""
    env_dir = client_dir / ".nodeenv"
    if (env_dir / "bin" / "npx").exists():
        rich.print("[bold](viser)[/bold] nodejs is set up!")
        return env_dir

    subprocess.run([sys.executable, "-m", "nodeenv", "--node=20.4.0", env_dir])
    subprocess.run(
        args=[env_dir / "bin" / "npm", "install", "yarn"],
        input="y\n".encode(),
    )
    assert (env_dir / "bin" / "npx").exists()
    return env_dir


def _modified_time_recursive(dir: Path) -> float:
    """Recursively get the last time a file was modified in a directory."""
    return max([f.stat().st_mtime for f in dir.glob("**/*")])
