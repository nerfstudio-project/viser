# TODO(by): work in progress...!

import sqlite3
from pathlib import Path

import tyro


def main(colmap_db: Path) -> None:
    sqlite3.connect(colmap_db)

    # COLMAP tables:
    # - cameras
    # - images
    # - keypoints
    # - descriptors
    # - matches
    # - two_view_geometries


if __name__ == "__main__":
    tyro.cli(main)
