import subprocess
import sys,os
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
        if os.name == "nt":
            print("Build on windows")
            env_dir = _install_sandboxed_node_win()
            env_dir_abs = env_dir / "Scripts"
            # 1. 构建 Windows 路径
            npx_path = env_dir_abs / "npx.cmd"
            yarn_install_cmd = f"{npx_path} yarn install"
            yarn_build_cmd = f"{npx_path} yarn run build"
            # 激活 Node 环境并运行 yarn 命令
            subprocess.run(["cmd", "/c", yarn_install_cmd], cwd=env_dir_abs, shell=True)
            print('install yarn done!!!!!!!')
            subprocess.run(["cmd", "/c", yarn_build_cmd], cwd=env_dir_abs, shell=True)
            print('build yarn done!!!!!!')
            # print('构建成功！！！！！！！！！！！！！！！！')
        else:
            print("Build on other platforms")
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

def _install_sandboxed_node_win() -> Path:
    """Install a sandboxed copy of nodejs using nodeenv, and return a path to the
    environment root."""
    # 1 Set the node environment path
    env_dir = client_dir / ".nodeenv"
    if (env_dir / "Scripts" / "npx.cmd").exists():
        rich.print("[bold](viser)[/bold] nodejs is set up!")
        return env_dir
    #######    Installing a node    you can run  in cmd :
    # Administrator permissions Open cmd
    # conda  activate your python
    # python -m nodeenv --node=20.4.0  path\to\your\extern\viser\src\viser\client\.nodeenv
    result = subprocess.run(
        [sys.executable, "-m", "nodeenv", "--node=20.4.0", env_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    try:
        # 首先尝试 UTF-8 编码
        error_output = result.stderr.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # 如果 UTF-8 失败，尝试使用 GBK 编码（适用于中文 Windows 系统）
            error_output = result.stderr.decode('gbk')
        except UnicodeDecodeError:
            # 如果其他编码也失败，使用 'replace' 选项替换无法解码的字符
            error_output = result.stderr.decode('utf-8', errors='replace')

    print(f"Link error: {error_output}")
    print('\n###############\n')
    print(f"About unable to create link error: Failed to create nodejs.exe link  ")
    print(
        " NOTE: This error does not affect the build process and can be fixed manually if necessary.")
    print(
        " 1. Open Command Prompt with Administrator privileges\n 2. Navigate to the .nodeenv\\Scripts directory in your project")
    print(
        " 3. Run the command 'mklink nodejs.exe node.exe' to create the symbolic link\n 4. After this, it should be okay")
    print('\n###############\n')

    print('node installation complete\n安装node完成！！！！！！！！！！！！！')
    subprocess.run(
        args=[env_dir / "Scripts" / "npm.cmd", "install", "yarn"],
        input="y\n".encode(),
    )
    print('install yarn done')
    assert (env_dir / "Scripts" / "npx.cmd").exists()
    return env_dir


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
