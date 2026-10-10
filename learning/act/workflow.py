#!/usr/bin/env python3
"""Unified ACT CLI. Commands are previews unless --execute is supplied."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
TRIAL_COLUMNS = ["trial_id", "cell", "position_set", "lighting", "episode_index", "grasp_success",
                 "place_success", "failure_direction", "notes"]


def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def flag(key, value):
    if isinstance(value, bool): value = str(value).lower()
    if value is None: value = "null"
    return f"--{key}={value}"


def checked_dataset(cfg):
    path = resolve(cfg["dataset"]["root"])
    info = json.loads((path / "meta/info.json").read_text())
    if not info.get("total_episodes", 0): raise ValueError("Dataset has no completed episodes")
    if info["fps"] != cfg["dataset"]["fps"]: raise ValueError("Dataset FPS differs from workflow config")
    expected = {"observation.state", "action", *[f"observation.images.{k}" for k in cfg["hardware"]["cameras"]]}
    missing = expected - set(info["features"])
    if missing: raise ValueError(f"Missing dataset features: {sorted(missing)}")
    return path, info


def checked_policy(path, cfg):
    p = resolve(path)
    for name in ("config.json", "model.safetensors"):
        if not (p / name).is_file(): raise FileNotFoundError(p / name)
    c = json.loads((p / "config.json").read_text())
    if c.get("type") != "act": raise ValueError("Checkpoint is not ACT")
    keys = {k for k in c["input_features"] if k.startswith("observation.images.")}
    expected = {f"observation.images.{k}" for k in cfg["hardware"]["cameras"]}
    if keys != expected: raise ValueError(f"Checkpoint cameras {keys} differ from config {expected}")
    for key in keys:
        camera = cfg["hardware"]["cameras"][key.split(".")[-1]]
        if c["input_features"][key]["shape"] != [3, camera["height"], camera["width"]]:
            raise ValueError(f"Resolution mismatch for {key}")
    return p


def record_flags(cfg, root, repo_id, episodes, seconds, reset):
    h = cfg["hardware"]
    return [flag("robot.type", "so101_follower"), flag("robot.id", h["follower_id"]),
            flag("robot.port", h["follower_port"]), flag("robot.max_relative_target", None),
            flag("robot.cameras", json.dumps(h["cameras"])), flag("dataset.repo_id", repo_id),
            flag("dataset.root", str(root)), flag("dataset.single_task", cfg["dataset"]["task"]),
            flag("dataset.fps", cfg["dataset"]["fps"]), flag("dataset.num_episodes", episodes),
            flag("dataset.episode_time_s", seconds), flag("dataset.reset_time_s", reset),
            flag("dataset.push_to_hub", False), flag("display_data", cfg["display_data"]),
            flag("play_sounds", cfg["play_sounds"])]


def summarize_trials(path):
    groups = {}
    seen = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(TRIAL_COLUMNS).issubset(reader.fieldnames or []): raise ValueError("Trial CSV schema mismatch")
        for row in reader:
            if row["trial_id"] in seen: raise ValueError(f"Duplicate trial ID: {row['trial_id']}")
            seen.add(row["trial_id"])
            if row["grasp_success"] not in ("0", "1"): raise ValueError("Every trial needs grasp_success 0 or 1")
            if row["place_success"] not in ("", "0", "1"): raise ValueError("place_success must be empty, 0 or 1")
            if row["place_success"] == "1" and row["grasp_success"] == "0": raise ValueError("Place success requires grasp success")
            key = (row["cell"], row["position_set"], row["lighting"])
            groups.setdefault(key, []).append(row)
    result = []
    for (cell, positions, lighting), rows in sorted(groups.items()):
        n = len(rows); k = sum(int(x["grasp_success"]) for x in rows); p = k/n; z = 1.96
        den = 1+z*z/n; mid=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        scored = [x for x in rows if x["place_success"] != ""]
        result.append({"cell": cell, "position_set": positions, "lighting": lighting, "trials": n,
                       "grasp_successes": k, "grasp_rate": p, "wilson95": [mid-half, mid+half],
                       "full_task_trials": len(scored),
                       "full_task_rate": sum(int(x["grasp_success"])*int(x["place_success"]) for x in scored)/len(scored) if scored else None})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "train", "resume", "deploy", "tensorboard", "trial-sheet", "evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs/act_workflow.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--append", action="store_true", help="Resume collection; episodes means additional episodes")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--run", help="New run name, or existing name for TensorBoard")
    parser.add_argument("--checkpoint", help="Exact pretrained_model directory")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--trial-file", type=Path)
    args=parser.parse_args(argv)
    if args.episodes is not None and args.episodes <= 0: raise ValueError("Episodes must be positive")
    if args.steps is not None and args.steps <= 0: raise ValueError("Steps must be positive")
    cfg=json.loads(args.config.read_text())
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    run=args.run or f"act_black15_{stamp}"
    if Path(run).name != run or run in (".", ".."): raise ValueError("Run name must be one directory component")
    run_dir=ROOT/"outputs/train"/run; tb=ROOT/"outputs/tensorboard"/run
    env=os.environ.copy(); commands=[]; destination=None
    if args.stage=="collect":
        dst=resolve(cfg["dataset"]["root"])
        if args.append: checked_dataset(cfg)
        elif dst.exists(): raise FileExistsError(f"Dataset exists; use --append or a new config root: {dst}")
        r=cfg["record"];h=cfg["hardware"]
        commands=[["lerobot-record",*record_flags(cfg,dst,cfg["dataset"]["repo_id"],args.episodes or r["num_episodes"],r["episode_time_s"],r["reset_time_s"]),
                   flag("teleop.type","so101_leader"),flag("teleop.id",h["leader_id"]),flag("teleop.port",h["leader_port"]),flag("resume",args.append)]]
    elif args.stage=="train":
        ds,info=checked_dataset(cfg); t=cfg["train"]
        if run_dir.exists(): raise FileExistsError(run_dir)
        options={"dataset.repo_id":cfg["dataset"]["repo_id"],"dataset.root":str(ds),"dataset.image_transforms.enable":t["image_augmentation"],
                 "policy.type":"act","policy.push_to_hub":False,"output_dir":str(run_dir),"job_name":run,
                 "batch_size":t["batch_size"],"steps":args.steps or t["steps"],"save_freq":t["save_freq"],"save_checkpoint":True,
                 "log_freq":t["log_freq"],"num_workers":t["num_workers"],"seed":t["seed"],"wandb.enable":False,
                 "policy.device":t["device"],"policy.chunk_size":t["chunk_size"],"policy.n_action_steps":t["n_action_steps"],
                 "policy.kl_weight":t["kl_weight"],"policy.temporal_ensemble_coeff":t["temporal_ensemble_coeff"],
                 "policy.optimizer_lr":t["optimizer_lr"],"policy.optimizer_lr_backbone":t["optimizer_lr"],
                 "policy.pretrained_backbone_weights":t["pretrained_backbone_weights"]}
        if t["temporal_ensemble_coeff"] is not None and t["n_action_steps"]!=1: raise ValueError("Temporal ensemble requires n_action_steps=1")
        commands=[["lerobot-train",*[flag(k,v) for k,v in options.items()]]]; env["LEROBOT_TENSORBOARD_DIR"]=str(tb)
    elif args.stage=="resume":
        if not args.checkpoint: raise ValueError("--checkpoint required")
        p=checked_policy(args.checkpoint,cfg); tc=p/"train_config.json"
        if not tc.is_file(): raise FileNotFoundError(tc)
        state=p.parent/"training_state"
        if not state.is_dir(): raise FileNotFoundError(f"Resume requires training state: {state}")
        saved=json.loads(tc.read_text()); steps=args.steps or saved["steps"]
        run_dir=resolve(saved["output_dir"]); tb=ROOT/"outputs/tensorboard"/run_dir.name
        commands=[["lerobot-train",flag("config_path",str(tc)),flag("resume",True),flag("steps",steps)]]
        env["LEROBOT_TENSORBOARD_DIR"]=str(tb)
    elif args.stage=="deploy":
        if not args.checkpoint: raise ValueError("--checkpoint required")
        p=checked_policy(args.checkpoint,cfg); r=cfg["deploy"]
        destination=ROOT/"outputs/evaluation"/run
        if destination.exists(): raise FileExistsError(destination)
        commands=[[sys.executable,str(ROOT/"learning/act/lerobot_record_log_actions.py"),"--action-log",str(destination/"actions.csv"),
                   *record_flags(cfg,destination/"dataset",f"local/eval_{run}",args.episodes or r["num_episodes"],r["episode_time_s"],r["reset_time_s"]),flag("policy.path",str(p))]]
    elif args.stage=="tensorboard":
        if not args.run: raise ValueError("--run required")
        commands=[["tensorboard","--logdir",str(tb),"--host","127.0.0.1","--port","6006"]]
    elif args.stage=="trial-sheet":
        if not args.trial_file: raise ValueError("--trial-file required")
        path=args.trial_file.expanduser().resolve()
        if path.exists(): raise FileExistsError(path)
        if args.execute:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x",newline="") as f:
                w=csv.writer(f);w.writerow(TRIAL_COLUMNS)
                for point in range(15):
                    for trial in range(args.episodes or 10):
                        w.writerow([f"P{point}_{trial+1:02d}",f"P{point}","seen","afternoon","","","","",""])
        print(f"{'Created' if args.execute else 'Would create'} trial sheet: {path}");return
    else:
        if not args.trial_file: raise ValueError("--trial-file required")
        print(json.dumps(summarize_trials(args.trial_file.expanduser().resolve()),indent=2));return
    for command in commands: print(shlex.join(command),flush=True)
    if "LEROBOT_TENSORBOARD_DIR" in env: print(f"TensorBoard: {tb}",flush=True)
    if not args.execute: print("Preview only. Add --execute to run.");return
    if args.stage in ("train","resume"):
        # Full metrics are available in the locally patched trainer. The log
        # converter provides basic loss/gradient/lr scalars on upstream installs.
        tb.mkdir(parents=True,exist_ok=True);log=tb/"train.log"
        (tb/f"workflow_{stamp}.json").write_text(json.dumps({"workflow":cfg,"command":commands[0],
            "output_dir":str(run_dir),"python":sys.version,"resumed":args.stage=="resume"},indent=2)+"\n")
        converter=[sys.executable,str(ROOT/"learning/act/lerobot_log_to_tensorboard.py"),"--logdir",str(tb/"log_scalars")]
        with log.open("a") as f:
            writer=subprocess.Popen(converter,stdin=subprocess.PIPE,text=True)
            trainer=None
            try:
                trainer=subprocess.Popen(commands[0],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
                for line in trainer.stdout:
                    f.write(line);f.flush()
                    writer.stdin.write(line);writer.stdin.flush()
                status=trainer.wait()
            finally:
                if trainer is not None and trainer.poll() is None:
                    trainer.terminate();trainer.wait()
                if writer.stdin: writer.stdin.close()
                writer.wait()
            if status: raise SystemExit(status)
    else:
        if destination:
            destination.mkdir(parents=True)
            (destination/"run_manifest.json").write_text(json.dumps({"checkpoint":str(p),"workflow":cfg,"command":commands[0]},indent=2)+"\n")
        subprocess.run(commands[0],env=env,check=True)


if __name__=="__main__":main()
