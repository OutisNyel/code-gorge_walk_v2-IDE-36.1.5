#!/usr/bin/env python3
"""Render conf/map_data/F_level_1.json into a 64x64 PNG map via OpenCV.

Rules:
- path: white
- obstacle: black
- start ([29, 9]): red
- end ([11, 55]): green
- state id encoding: id = x * 64 + y
- direction mapping: 0=up, 1=down, 2=left, 3=right
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MAP_SIZE = 64
DEFAULT_START = (29, 9)
DEFAULT_END = (11, 55)

# OpenCV uses BGR color order.
COLOR_OBSTACLE = (0, 0, 0)
COLOR_PATH = (255, 255, 255)
COLOR_START = (0, 0, 255)
COLOR_END = (0, 255, 0)

# 0-3 directions: up, down, left, right
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


def encode_position(x: int, y: int) -> int:
    return x * MAP_SIZE + y


def decode_state_id(state_id: int) -> tuple[int, int]:
    return state_id // MAP_SIZE, state_id % MAP_SIZE


def collect_path_cells(map_data: dict) -> set[int]:
    """Collect traversable cells from 0/1/2/3 transitions (up/down/left/right)."""
    path_cells: set[int] = set()

    for state_key, action_map in map_data.items():
        state_id = int(state_key)
        path_cells.add(state_id)

        if not isinstance(action_map, dict):
            continue

        # Parse transitions explicitly in action index order: 0,1,2,3.
        for action in range(4):
            transition = action_map.get(str(action))
            if not (isinstance(transition, list) and len(transition) >= 1):
                continue

            next_state = int(transition[0])
            path_cells.add(next_state)

    return path_cells


def render_png(path_cells: set[int], output_path: Path, start: tuple[int, int], end: tuple[int, int], scale: int) -> None:
    img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    img[:, :] = COLOR_OBSTACLE

    for state_id in path_cells:
        x, y = decode_state_id(state_id)
        if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
            img[x, y] = COLOR_PATH

    img[start[0], start[1]] = COLOR_START
    img[end[0], end[1]] = COLOR_END

    if scale > 1:
        img = cv2.resize(img, (MAP_SIZE * scale, MAP_SIZE * scale), interpolation=cv2.INTER_NEAREST)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render F_level_1.json to a 64x64 PNG map image.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("conf/map_data/F_level_1.json"),
        help="Path to F_level_1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("My/f_level_1_map.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="Pixels per cell (default: 8)",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=DEFAULT_START,
        help="Start coordinate x y (default: 29 9)",
    )
    parser.add_argument(
        "--end",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=DEFAULT_END,
        help="End coordinate x y (default: 11 55)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    with args.input.open("r", encoding="utf-8") as f:
        map_data = json.load(f)

    start = (int(args.start[0]), int(args.start[1]))
    end = (int(args.end[0]), int(args.end[1]))
    path_cells = collect_path_cells(map_data)

    render_png(path_cells, args.output, start, end, args.scale)
    print(f"Rendered map to: {args.output}")
    print(f"Path cell count: {len(path_cells)}")
    print(f"Start: {start}, End: {end}")


if __name__ == "__main__":
    main()
