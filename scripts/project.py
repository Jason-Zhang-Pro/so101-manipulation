#!/usr/bin/env python3
"""Select an explicit runtime profile. Preview is dependency-free."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_plan(profile_name, task, extra=(), *, execute=False):
    manifest = json.loads((ROOT / "configs/runtime_profiles.json").read_text(encoding="utf-8"))
    profile = manifest["profiles"][profile_name]
    if task not in profile["commands"]:
        raise ValueError(f"Task {task!r} is unavailable in profile {profile_name!r}; choose {list(profile['commands'])}")
    if any(arg == "--execute" or arg.startswith("--execute=") for arg in extra):
        raise ValueError("Use the launcher's --execute before --, not in forwarded arguments.")
    command = [sys.executable if part == "{python}" else part for part in profile["commands"][task]]
    if profile_name == "primary" and task != "sim-act":
        command += ["--config", str(ROOT / profile["workflow_config"])]
    command += list(extra)
    if execute and (task in {"collect", "train", "resume", "deploy", "evaluate"}):
        command.append("--execute")
    return {"profile": profile_name, "task": task, "settings": profile,
            "policy_contract": manifest["policy_contract"], "command": command,
            "cwd": str(ROOT), "execute": execute}


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in values:
        split = values.index("--")
        values, extra = values[:split], values[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=["primary", "capstone_ppo"])
    parser.add_argument("task")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(values)
    try:
        plan = make_plan(args.profile, args.task, extra, execute=args.execute)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2) if args.json else
          f"Profile: {args.profile}; workspace: {plan['settings']['ros_workspace']}\n" + shlex.join(plan["command"]), flush=True)
    if not args.execute:
        print("Preview only. Add --execute before -- to run.")
        return 0
    # Two same-named ROS description packages must never be overlaid together.
    other = "ros2_ws" if args.profile == "primary" else "ros"
    other_install = (ROOT / other / "install").resolve()
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix and Path(prefix).resolve().is_relative_to(other_install):
            parser.error(f"Conflicting ROS workspace is sourced: {other_install}. Use a fresh terminal.")
    env = os.environ.copy()
    sources = [str(ROOT / "src"), str(ROOT / "lerobot/src")]
    env["PYTHONPATH"] = os.pathsep.join(sources + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["LEROBOT_SRC"] = str(ROOT / "lerobot/src")
    env["PROJECT_ROOT"] = str(ROOT)
    return subprocess.call(plan["command"], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
