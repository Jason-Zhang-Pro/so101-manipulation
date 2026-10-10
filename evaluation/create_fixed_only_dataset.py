#!/usr/bin/env python3
"""Create a LeRobot dataset view containing only the fixed camera."""

import argparse
import json
import os
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if dst.exists():
        raise FileExistsError(dst)

    info = json.loads((src / "meta/info.json").read_text())
    fixed_key = "observation.images.fixed"
    handeye_key = "observation.images.handeye"
    if fixed_key not in info["features"] or handeye_key not in info["features"]:
        raise RuntimeError("Expected fixed and handeye image features")

    (dst / "meta").mkdir(parents=True)
    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "videos/chunk-000" / fixed_key).mkdir(parents=True)

    info["features"].pop(handeye_key)
    info["total_videos"] = info["total_episodes"]
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    shutil.copy2(src / "meta/episodes.jsonl", dst / "meta/episodes.jsonl")
    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")

    episode_stats = read_jsonl(src / "meta/episodes_stats.jsonl")
    for row in episode_stats:
        row["stats"].pop(handeye_key, None)
    write_jsonl(dst / "meta/episodes_stats.jsonl", episode_stats)

    for parquet in sorted((src / "data/chunk-000").glob("episode_*.parquet")):
        os.link(parquet, dst / "data/chunk-000" / parquet.name)
    for video in sorted((src / "videos/chunk-000" / fixed_key).glob("episode_*.mp4")):
        os.link(video, dst / "videos/chunk-000" / fixed_key / video.name)

    summary = {
        "source": str(src),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "image_features": [fixed_key],
        "storage": "hardlinks",
    }
    (dst / "fixed_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
