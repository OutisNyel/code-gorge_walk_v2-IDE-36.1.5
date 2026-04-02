#!/usr/bin/env python3
"""Render a Kaiwu transition-map JSON file into a PNG image via OpenCV.

State encoding contract:
    state_id = x * map_size + y
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_INPUT = Path("conf/map_data/F_level_1.json")
DEFAULT_ENV_CONF = Path("agent_q_learning/conf/train_env_conf.toml")
DEFAULT_MAP_SIZE = 64
DEFAULT_SCALE = 8

# OpenCV uses BGR color order.
DEFAULT_COLOR_OBSTACLE = (0, 0, 0)
DEFAULT_COLOR_PATH = (255, 255, 255)
DEFAULT_COLOR_START = (0, 0, 255)
DEFAULT_COLOR_END = (0, 255, 0)
DEFAULT_COLOR_TREASURE = (0, 255, 255)
DEFAULT_COLOR_TERMINAL = (255, 0, 255)


def decode_state_id(state_id: int, map_size: int) -> tuple[int, int]:
    return divmod(state_id, map_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Kaiwu transition-map JSON into a PNG map image.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input transition-map JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: My/<input_stem>_map.png).",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=DEFAULT_MAP_SIZE,
        help="Map side length used for state decoding (default: 64).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="Pixels per cell (default: 8).",
    )
    parser.add_argument(
        "--env-conf",
        type=Path,
        default=DEFAULT_ENV_CONF,
        help="TOML path used for default start/end when --start/--end are omitted.",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional start coordinate override.",
    )
    parser.add_argument(
        "--end",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional end coordinate override.",
    )
    parser.add_argument(
        "--color-obstacle",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_OBSTACLE,
        help="Obstacle color in BGR (default: 0 0 0).",
    )
    parser.add_argument(
        "--color-path",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_PATH,
        help="Walkable path color in BGR (default: 255 255 255).",
    )
    parser.add_argument(
        "--color-start",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_START,
        help="Start color in BGR (default: 0 0 255).",
    )
    parser.add_argument(
        "--color-end",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_END,
        help="End color in BGR (default: 0 255 0).",
    )
    parser.add_argument(
        "--color-treasure",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_TREASURE,
        help="Treasure transition color in BGR (default: 0 255 255).",
    )
    parser.add_argument(
        "--color-terminal",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=DEFAULT_COLOR_TERMINAL,
        help="Terminal transition color in BGR (default: 255 0 255).",
    )
    return parser.parse_args()


def ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def parse_color(name: str, raw_color: Any) -> tuple[int, int, int]:
    if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        raise ValueError(f"{name} must be a BGR triplet.")

    color = tuple(int(v) for v in raw_color)
    for channel in color:
        if channel < 0 or channel > 255:
            raise ValueError(f"{name} channel out of range [0, 255]: {color}")
    return color


def parse_coordinate(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [x, y].")
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError(f"{label} values must be integers.")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError(f"{label} values must be integers.")
    return x, y


def validate_coordinate(coord: tuple[int, int], label: str, map_size: int) -> None:
    x, y = coord
    if not (0 <= x < map_size and 0 <= y < map_size):
        raise ValueError(f"{label} out of range for map_size={map_size}: {coord}")


def load_map_data(input_path: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input map file does not exist: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {input_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Map JSON root must be an object: {state_id: action_map}.")

    return data


def load_start_end_from_env_conf(env_conf_path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    if not env_conf_path.is_file():
        raise FileNotFoundError(f"env-conf file does not exist: {env_conf_path}")

    try:
        conf_text = env_conf_path.read_text(encoding="utf-8")
        conf = tomllib.loads(conf_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {env_conf_path}: {exc}") from exc

    env_conf = conf.get("env_conf")
    if not isinstance(env_conf, dict):
        raise ValueError(f"Missing [env_conf] section in {env_conf_path}")

    if "start" not in env_conf or "end" not in env_conf:
        raise ValueError(f"env_conf.start and env_conf.end are required in {env_conf_path}")

    start = parse_coordinate(env_conf["start"], f"{env_conf_path}:env_conf.start")
    end = parse_coordinate(env_conf["end"], f"{env_conf_path}:env_conf.end")
    return start, end


def validate_state_id(state_id: int, map_size: int, context: str) -> None:
    if isinstance(state_id, bool) or not isinstance(state_id, int):
        raise ValueError(f"{context}: state id must be an integer, got {state_id!r}")
    if state_id < 0:
        raise ValueError(f"{context}: state id must be non-negative, got {state_id}")

    x, y = decode_state_id(state_id, map_size)
    if not (0 <= x < map_size and 0 <= y < map_size):
        raise ValueError(
            f"{context}: state id {state_id} decodes to {(x, y)} outside map_size={map_size}"
        )


def parse_transition(transition: Any, state_key: str, action_key: str) -> tuple[int, float, bool]:
    context = f"state {state_key}, action {action_key}"
    if not isinstance(transition, list) or len(transition) < 3:
        raise ValueError(f"{context}: transition must be [next_state, reward, done].")

    next_state = transition[0]
    reward = transition[1]
    done = transition[2]

    if isinstance(next_state, bool) or not isinstance(next_state, int):
        raise ValueError(f"{context}: next_state must be an integer, got {next_state!r}")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError(f"{context}: reward must be numeric, got {reward!r}")
    if not isinstance(done, bool):
        raise ValueError(f"{context}: done must be bool, got {done!r}")

    return int(next_state), float(reward), done


def collect_semantic_cells(
    map_data: dict[str, Any],
    map_size: int,
) -> tuple[set[int], set[int], set[int]]:
    walkable_cells: set[int] = set()
    treasure_cells: set[int] = set()
    terminal_cells: set[int] = set()

    for state_key, action_map in map_data.items():
        try:
            state_id = int(state_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"State key must be integer-like, got {state_key!r}") from exc

        validate_state_id(state_id, map_size, f"source state {state_key}")
        walkable_cells.add(state_id)

        if not isinstance(action_map, dict):
            raise ValueError(f"State {state_key}: action map must be an object.")

        for action in range(4):
            action_key = str(action)
            if action_key not in action_map:
                raise ValueError(f"State {state_key}: missing action key {action_key}.")

            next_state, reward, done = parse_transition(action_map[action_key], state_key, action_key)
            validate_state_id(next_state, map_size, f"state {state_key}, action {action_key}")

            walkable_cells.add(next_state)
            if done:
                terminal_cells.add(next_state)
            elif reward > 0:
                treasure_cells.add(next_state)

    return walkable_cells, treasure_cells, terminal_cells


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return Path("My") / f"{input_path.stem}_map.png"


def paint_cells(
    image: np.ndarray,
    state_ids: set[int],
    map_size: int,
    color: tuple[int, int, int],
) -> None:
    for state_id in state_ids:
        x, y = decode_state_id(state_id, map_size)
        image[x, y] = color


def render_png(
    output_path: Path,
    map_size: int,
    scale: int,
    start: tuple[int, int],
    end: tuple[int, int],
    walkable_cells: set[int],
    treasure_cells: set[int],
    terminal_cells: set[int],
    color_obstacle: tuple[int, int, int],
    color_path: tuple[int, int, int],
    color_start: tuple[int, int, int],
    color_end: tuple[int, int, int],
    color_treasure: tuple[int, int, int],
    color_terminal: tuple[int, int, int],
) -> None:
    image = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    image[:, :] = color_obstacle

    # Precedence from low to high:
    # obstacle -> path -> treasure -> terminal -> start/end
    paint_cells(image, walkable_cells, map_size, color_path)
    paint_cells(image, treasure_cells, map_size, color_treasure)
    paint_cells(image, terminal_cells, map_size, color_terminal)
    image[start[0], start[1]] = color_start
    image[end[0], end[1]] = color_end

    if scale > 1:
        image = cv2.resize(
            image,
            (map_size * scale, map_size * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")


def main() -> None:
    args = parse_args()

    ensure_positive("--map-size", args.map_size)
    ensure_positive("--scale", args.scale)

    color_obstacle = parse_color("--color-obstacle", args.color_obstacle)
    color_path = parse_color("--color-path", args.color_path)
    color_start = parse_color("--color-start", args.color_start)
    color_end = parse_color("--color-end", args.color_end)
    color_treasure = parse_color("--color-treasure", args.color_treasure)
    color_terminal = parse_color("--color-terminal", args.color_terminal)

    map_data = load_map_data(args.input)
    walkable_cells, treasure_cells, terminal_cells = collect_semantic_cells(map_data, args.map_size)

    env_start: tuple[int, int] | None = None
    env_end: tuple[int, int] | None = None
    if args.start is None or args.end is None:
        env_start, env_end = load_start_end_from_env_conf(args.env_conf)

    if args.start is None:
        if env_start is None:
            raise RuntimeError("env_conf start was not loaded.")
        start = env_start
        start_source = f"env-conf:{args.env_conf}"
    else:
        start = parse_coordinate(args.start, "--start")
        start_source = "cli"

    if args.end is None:
        if env_end is None:
            raise RuntimeError("env_conf end was not loaded.")
        end = env_end
        end_source = f"env-conf:{args.env_conf}"
    else:
        end = parse_coordinate(args.end, "--end")
        end_source = "cli"

    validate_coordinate(start, "start", args.map_size)
    validate_coordinate(end, "end", args.map_size)
    if start == end:
        raise ValueError(f"start and end must be different, both were {start}")

    output_path = resolve_output_path(args.input, args.output)

    render_png(
        output_path=output_path,
        map_size=args.map_size,
        scale=args.scale,
        start=start,
        end=end,
        walkable_cells=walkable_cells,
        treasure_cells=treasure_cells,
        terminal_cells=terminal_cells,
        color_obstacle=color_obstacle,
        color_path=color_path,
        color_start=color_start,
        color_end=color_end,
        color_treasure=color_treasure,
        color_terminal=color_terminal,
    )

    print(f"Rendered map to: {output_path}")
    print(f"Map size: {args.map_size}x{args.map_size}")
    print(f"Scale: {args.scale}")
    print(
        "Cell counts: "
        f"walkable={len(walkable_cells)}, "
        f"treasure={len(treasure_cells)}, "
        f"terminal={len(terminal_cells)}"
    )
    print(f"Start: {start} (source={start_source})")
    print(f"End: {end} (source={end_source})")


if __name__ == "__main__":
    main()
