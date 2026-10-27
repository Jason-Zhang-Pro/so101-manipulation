"""Offline audit of ACT units, recorded PPO samples and gripper tracking.

No Gazebo, training, policy changes or process control. Stored ``positions``
are observations before that row's action; tracking compares to the NEXT row.
This is a mapping/command audit, not a deterministic policy performance eval.
Run with PYTHONPATH=src using the project's Python environment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from so101_capstone.policies.so101_sim_calibration import (
    SIM_CALIBRATION_FILE,
    SIM_JOINT_CALIBRATIONS,
    SIM_POLICY_JOINT_NAMES,
    gazebo_positions_to_policy,
    policy_positions_to_gazebo,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = (
    ROOT / "output_simulation/finetuned/act_ppo_20260928_200236_from_checkpoints",
    ROOT / "output_simulation/finetuned/act_ppo_20260929_160417_resume_update25",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    # Paused runs should be complete. Reject malformed records, not silently
    # accept an incomplete/partially written audit input.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    assert np.isfinite(a).all()
    return {"count": int(a.size), "mean": float(a.mean()),
            "rms": float(np.sqrt(np.mean(a * a))), "min": float(a.min()),
            "max": float(a.max()), "mean_absolute": float(np.abs(a).mean())}


def audit_calibration(reward: dict, stages: dict) -> dict:
    home = np.asarray(reward["home_joint_positions_rad"])
    lower = np.asarray([c.policy_lower for c in SIM_JOINT_CALIBRATIONS])
    upper = np.asarray([c.policy_upper for c in SIM_JOINT_CALIBRATIONS])
    errors = []
    for fraction in np.linspace(0, 1, 101):
        policy = lower + fraction * (upper - lower)
        actual = gazebo_positions_to_policy(policy_positions_to_gazebo(policy))
        errors.append(float(np.max(np.abs(actual - policy))))
    assert max(errors) < 1e-10
    home_policy = gazebo_positions_to_policy(home, clip=False)
    assert np.all(home_policy >= lower) and np.all(home_policy <= upper)
    grip = SIM_JOINT_CALIBRATIONS[-1]
    table = []
    for name, q in (
        ("calibrated_lower", grip.gazebo_lower_rad),
        ("HOME", float(home[-1])),
        ("S1_opening_min", float(home[-1] + stages["s1_opening_min_rad"])),
        ("S1_opening_max", float(home[-1] + stages["s1_opening_max_rad"])),
        ("calibrated_upper", grip.gazebo_upper_rad),
    ):
        table.append({"name": name, "policy_units": grip.gazebo_to_policy(q, clip=False),
                      "absolute_joint_rad": q, "absolute_joint_deg": float(np.degrees(q)),
                      "opening_relative_to_HOME_rad": q - float(home[-1])})
    return {"file": str(SIM_CALIBRATION_FILE), "sha256": digest(SIM_CALIBRATION_FILE),
            "joint_order": list(SIM_POLICY_JOINT_NAMES),
            "policy_lower": lower.tolist(), "policy_upper": upper.tolist(),
            "home_rad": home.tolist(), "home_policy_units": home_policy.tolist(),
            "gripper_slope_rad_per_policy_unit":
                (grip.gazebo_upper_rad - grip.gazebo_lower_rad) / 100,
            "gripper_intercept_rad": grip.gazebo_lower_rad,
            "roundtrip_max_abs_error_policy_units": max(errors), "gripper_table": table}


def audit_run(run: Path) -> dict:
    config = json.loads((run / "config.json").read_text())
    reward = json.loads((run / "resolved_reward.json").read_text())
    stages = json.loads((run / "resolved_stages.json").read_text())
    # Load recorded policy version using update event boundaries. Resumed run
    # begins with the checkpoint version, not zero.
    resume = config["gazebo"]["resume_checkpoint"]
    version = int(Path(resume).parent.name.removeprefix("update_")) if resume else 0
    episodes = []
    updates = []
    for row in read_jsonl(run / "metrics.jsonl"):
        if row["type"] == "ppo_update":
            version = row["update_index"]
            updates.append(version)
        elif row["type"] == "episode" and row.get("valid", True):
            episodes.append((version, row))
    lower = np.asarray([c.policy_lower for c in SIM_JOINT_CALIBRATIONS])
    upper = np.asarray([c.policy_upper for c in SIM_JOINT_CALIBRATIONS])
    qlower = np.asarray([c.gazebo_lower_rad for c in SIM_JOINT_CALIBRATIONS])
    qupper = np.asarray([c.gazebo_upper_rad for c in SIM_JOINT_CALIBRATIONS])
    centre, half_range = (upper + lower) / 2, (upper - lower) / 2
    home = np.asarray(reward["home_joint_positions_rad"])
    state_error = np.zeros(6)
    action_error = np.zeros(6)
    stats = defaultdict(lambda: defaultdict(list))
    traces = []
    total_steps = 0
    mean_out_of_bounds = 0
    clipped_gripper_samples = 0
    for version, episode in episodes:
        path = run / "episode_steps" / f"episode_{episode['episode_index']:06d}.jsonl"
        rows = read_jsonl(path)
        assert len(rows) == episode["step_count"], str(path)
        stage = episode["training_stage"]
        positions = np.asarray([r["positions"] for r in rows])
        states = np.asarray([r["actor_policy_state"] for r in rows])
        means = np.asarray([r["policy_action_means"] for r in rows])
        samples = np.asarray([r["policy_actions"] for r in rows])
        latents = np.asarray([r["policy_action_latents"] for r in rows])
        assert all(np.isfinite(a).all() for a in (positions, states, means, samples, latents))
        mapped_states = lower + (np.clip(positions, qlower, qupper) - qlower) / (qupper - qlower) * (upper - lower)
        state_error = np.maximum(state_error, np.abs(mapped_states - states).max(axis=0))
        reconstructed_samples = centre + half_range * np.clip(latents, -1, 1)
        action_error = np.maximum(action_error, np.abs(samples - reconstructed_samples).max(axis=0))
        mean_q = qlower + (np.clip(means, lower, upper) - lower) / (upper - lower) * (qupper - qlower)
        sample_q = qlower + (samples - lower) / (upper - lower) * (qupper - qlower)
        mean_out_of_bounds += int(np.count_nonzero((means[:, -1] < 0) | (means[:, -1] > 100)))
        clipped_gripper_samples += int(np.count_nonzero((latents[:, -1] < -1) | (latents[:, -1] > 1)))
        total_steps += len(rows)
        group = stats[f"policy_{version:06d}_{stage}"]
        group["episode_index"].append(episode["episode_index"])
        # Last second avoids treating the commanded opening transient after
        # HOME reset as a steady-state controller tracking defect.
        tail = slice(max(0, len(rows) - 30), len(rows) - 1)
        group["tail_mean_gripper_policy_units"].append(float(means[tail, -1].mean()))
        group["tail_mean_command_gripper_rad"].append(float(mean_q[tail, -1].mean()))
        group["tail_measured_gripper_rad"].append(float(positions[1:][tail, -1].mean()))
        group["tail_gripper_HOME_error_rad"].append(float(np.abs(positions[1:][tail, -1] - home[-1]).mean()))
        group["tail_tracking_error_next_observation_minus_command_rad"].extend(
            (positions[1:, -1] - sample_q[:-1, -1])[tail].tolist())
        group["tail_sample_minus_mean_command_rad"].extend(
            (sample_q[:, -1] - mean_q[:, -1])[tail].tolist())
        if stage == "S0" and (not traces or version == max(v for v, _ in episodes)):
            trace = {"episode_index": episode["episode_index"], "policy_update": version,
                     "file": str(path), "points": []}
            for index in (0, 9, 29, len(rows) - 2):
                trace["points"].append({"step": rows[index]["step_index"],
                    "gripper_before_rad": float(positions[index, -1]),
                    "mean_policy_units": float(means[index, -1]),
                    "sample_policy_units": float(samples[index, -1]),
                    "mean_command_rad": float(mean_q[index, -1]),
                    "sample_command_rad": float(sample_q[index, -1]),
                    "next_observed_rad": float(positions[index + 1, -1])})
            traces.append(trace)
    assert float(state_error.max()) < 3e-5, ("state mapping mismatch", state_error)
    assert float(action_error.max()) < 2e-5, ("action reconstruction mismatch", action_error)
    grouped = {name: {key: (values if key == "episode_index" else summarize(values))
                     for key, values in group.items()} for name, group in stats.items()}
    return {"run": str(run), "metrics_sha256": digest(run / "metrics.jsonl"),
            "calibration": audit_calibration(reward, stages),
            "completed_updates": updates, "audited_episodes": len(episodes),
            "audited_steps": total_steps, "state_mapping_max_error_per_joint": state_error.tolist(),
            "sample_reconstruction_max_error_per_joint": action_error.tolist(),
            "gripper_out_of_bounds_means": mean_out_of_bounds,
            "gripper_clipped_samples": clipped_gripper_samples,
            "groups": grouped, "S0_traces": traces}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "codex_workspace/reports/2026-09-29_gripper_action_contract.json")
    args = parser.parse_args()
    result = {"scope": "offline mapping and recorded control audit; no training or simulation",
              "tracking_note": "next row observations, not interpolated task-evidence poses; dynamic lag is expected",
              "runs": [audit_run(run.resolve()) for run in (args.run or DEFAULT_RUNS)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for run in result["runs"]:
        print(Path(run["run"]).name, run["audited_episodes"], "episodes", run["audited_steps"], "steps",
              "state/sample max error", max(run["state_mapping_max_error_per_joint"]),
              max(run["sample_reconstruction_max_error_per_joint"]))
    print(args.output)


if __name__ == "__main__":
    main()
