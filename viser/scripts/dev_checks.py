#!/usr/bin/env python
"""Runs formatting, linting, and type checking tests."""
import subprocess
import sys

import tyro
from rich.style import Style

from rich import console

CONSOLE = console.Console()

TYPE_TESTS = ["pyright .", "mypy ."]
FORMAT_TESTS = ["black .", "ruff --fix ."]


def run_command(command: str, continue_on_fail: bool = False) -> bool:
    """Run a command kill actions if it fails

    Args:
        command: command to run
        continue_on_fail: whether to continue running commands if the current one fails.
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        CONSOLE.print(f"[bold red]Error: `{command}` failed.")
        if not continue_on_fail:
            sys.exit(1)
    return ret_code == 0


def run_code_checks(
    continue_on_fail: bool = False,
    skip_format_checks: bool = False,
    skip_type_checks: bool = False,
):
    """Runs formatting, linting, and type checking tests.

    Args:
        continue_on_fail: Whether or not to continue running actions commands if the current one fails
        skip_format_checks: Whether or not to skip format tests
        skip_type_checks: Whether or not to skip type tests
    """

    success = True

    assert (
        not skip_format_checks or not skip_type_checks
    ), "Cannot skip format and type tests at the same time"
    tests = []
    if not skip_format_checks:
        tests += FORMAT_TESTS
    if not skip_type_checks:
        tests += TYPE_TESTS

    for test in tests:
        CONSOLE.line()
        CONSOLE.rule(f"[bold green]Running: {test}")
        success = success and run_command(test, continue_on_fail=continue_on_fail)

    if success:
        CONSOLE.line()
        CONSOLE.rule(characters="=")
        CONSOLE.print(
            "[bold green]:TADA: :TADA: :TADA: ALL CHECKS PASSED :TADA: :TADA: :TADA:",
            justify="center",
        )
        CONSOLE.rule(characters="=")
    else:
        CONSOLE.line()
        CONSOLE.rule(characters="=", style=Style(color="red"))
        CONSOLE.print(
            "[bold red]:skull: :skull: :skull: ERRORS FOUND :skull: :skull: :skull:",
            justify="center",
        )
        CONSOLE.rule(characters="=", style=Style(color="red"))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.cli(run_code_checks)


if __name__ == "__main__":
    entrypoint()
