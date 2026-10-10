#!/usr/bin/env python3
"""Run lerobot-record while saving policy actions and observed joint positions.

This keeps the upstream LeRobot source untouched. Use it exactly like
`lerobot-record`, with a few optional logging flags before/among the usual args:

    python tools/lerobot_record_log_actions.py --action-log /tmp/actions.csv --log-first-n 30 ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-log", default="/tmp/lerobot_policy_first_actions.csv")
    parser.add_argument("--log-first-n", type=int, default=10000)
    parser.add_argument("--log-every-n", type=int, default=1)
    parser.add_argument("--append-action-log", action="store_true")
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper_args, lerobot_args = parse_wrapper_args(sys.argv[1:])

    import lerobot.record as record_mod

    original_predict_action = record_mod.predict_action
    log_path = Path(wrapper_args.action_log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if wrapper_args.append_action_log else "w"
    log_file = log_path.open(mode, newline="")
    writer = csv.writer(log_file)
    if mode == "w" or log_path.stat().st_size == 0:
        writer.writerow(
            ["wall_time", "policy_call", "inference_ms"]
            + [f"target.{name}" for name in ACTION_NAMES]
            + [f"qpos.{name}" for name in ACTION_NAMES]
        )
        log_file.flush()

    state = {"policy_call": 0}

    def wrapped_predict_action(*args, **kwargs):
        started = time.perf_counter()
        action_values = original_predict_action(*args, **kwargs)
        inference_ms = (time.perf_counter() - started) * 1000
        call_idx = state["policy_call"]
        state["policy_call"] += 1

        if call_idx < wrapper_args.log_first_n and call_idx % max(wrapper_args.log_every_n, 1) == 0:
            values = action_values.detach().cpu().flatten().tolist()
            row_values = values[: len(ACTION_NAMES)]
            if len(row_values) < len(ACTION_NAMES):
                row_values.extend([""] * (len(ACTION_NAMES) - len(row_values)))
            observation = args[0] if args else kwargs.get("observation", {})
            qpos = observation.get("observation.state", [])
            if hasattr(qpos, "detach"):
                qpos = qpos.detach().cpu().flatten().tolist()
            elif hasattr(qpos, "flatten"):
                qpos = qpos.flatten().tolist()
            else:
                qpos = list(qpos)
            qpos = qpos[: len(ACTION_NAMES)]
            qpos.extend([""] * (len(ACTION_NAMES) - len(qpos)))
            writer.writerow(
                [f"{time.time():.6f}", call_idx, f"{inference_ms:.3f}"]
                + [f"{v:.6f}" for v in row_values]
                + [f"{v:.6f}" if v != "" else "" for v in qpos]
            )
            log_file.flush()
            print(
                "[ACT_ACTION_LOG]",
                f"step={call_idx}",
                f"inference_ms={inference_ms:.1f}",
                " ".join(f"{name}={value:.3f}" for name, value in zip(ACTION_NAMES, values, strict=False)),
                flush=True,
            )

        return action_values

    record_mod.predict_action = wrapped_predict_action
    sys.argv = ["lerobot-record", *lerobot_args]
    try:
        record_mod.main()
    finally:
        log_file.close()
        print(f"[ACT_ACTION_LOG] saved {log_path}", flush=True)


if __name__ == "__main__":
    main()
