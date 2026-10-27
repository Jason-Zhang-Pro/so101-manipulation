from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "analyze_act_rollouts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_act_rollouts",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_episode(path: Path, *, steps: int = 10) -> None:
    positions = np.zeros((steps, 6), dtype=np.float64)
    actions = positions.copy()
    actions[:, 0] = 0.03
    actions[:, 5] = np.linspace(0.0, 0.09, steps)
    next_positions = actions * 0.8
    rewards = np.linspace(-0.1, 0.1, steps)
    infos = []

    for step in range(steps):
        infos.append(
            json.dumps(
                {
                    "distance_m": None if step < 6 else 0.3 - 0.01 * (step - 6),
                    "camera_pair_reused": step in {1, 2, 5},
                    "consecutive_camera_reuse_steps": (
                        2 if step == 2 else 1 if step in {1, 5} else 0
                    ),
                    "camera_lag_from_step_end_s": 0.05,
                    "gripper_open_norm": step / max(1, steps - 1),
                    "success_ready_now": False,
                    "success": False,
                }
            )
        )

    np.savez_compressed(
        path,
        step_index=np.arange(steps, dtype=np.int64),
        observation_joint_positions=positions,
        observation_joint_velocities=np.zeros_like(positions),
        next_observation_joint_positions=next_positions,
        next_observation_joint_velocities=np.zeros_like(positions),
        action_joint_targets=actions,
        reward=rewards,
        terminated=np.zeros(steps, dtype=bool),
        truncated=np.zeros(steps, dtype=bool),
        done=np.zeros(steps, dtype=bool),
        info_json=np.asarray(infos),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "joint_order": list(module.DEFAULT_JOINT_NAMES),
                    "max_joint_delta_rad": 0.03,
                }
            )
        ),
    )


def test_load_and_analyze_schema_tolerant_episode(tmp_path):
    path = tmp_path / "episode_000000.npz"
    write_episode(path)
    episode = module.load_episode_archive(path)
    report = module.analyze_episode(
        episode,
        target_delay_steps=6,
        max_joint_delta_rad=0.03,
        joint_names=module.DEFAULT_JOINT_NAMES,
        unchanged_tolerance=1e-4,
        comparison_tolerance=1e-8,
    )

    assert report["step_count"] == 10
    assert report["resolved_sources"]["executed_action"] == (
        "action_joint_targets"
    )
    clipping = report["max_joint_delta_clipping"]
    assert clipping["method"] == "executed_delta_boundary_inference"
    assert clipping["per_joint"]["shoulder_pan"]["count"] == 10
    assert clipping["per_joint"]["shoulder_pan"]["ratio"] == pytest.approx(1.0)
    assert report["camera"]["reuse_count"] == 3
    assert report["camera"]["reported_max_consecutive_reuse"] == 2
    assert report["phase_comparison"]["pre_target"]["step_count"] == 6
    assert report["phase_comparison"]["post_target"]["step_count"] == 4
    assert report["optional_fields"]["distance_m"] is not None


def test_analyze_paths_writes_reports(tmp_path):
    path = tmp_path / "episode_000001.npz"
    output = tmp_path / "diagnostics"
    write_episode(path, steps=8)

    combined = module.analyze_paths(
        [path],
        output_dir=output,
        target_delay_steps=5,
        max_joint_delta_rad=0.03,
        make_plots=False,
    )

    assert combined["episode_count"] == 1
    assert (output / "episode_000001_diagnostic.txt").is_file()
    assert (output / "episode_000001_diagnostic.json").is_file()
    assert (output / "episode_000001_curves.csv").is_file()
    assert (output / "combined_diagnostic.json").is_file()


def test_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_episode_archive(tmp_path / "missing.npz")


def test_camera_age_uses_archived_fresh_and_reused_frame_limits(tmp_path):
    from dataclasses import replace

    path = tmp_path / "episode.npz"
    write_episode(path, steps=4)
    episode = replace(module.load_episode_archive(path), per_step_series={
        "camera_lag_from_step_end_s": np.array([0.27, 0.31, 0.04, 0.06]),
        "camera_pair_reused": np.array([True, True, False, False]),
        "max_camera_reuse_lag_s": np.full(4, 0.30),
        "max_camera_lag_from_step_end_s": np.full(4, 0.05),
    })
    report = module._camera_report(episode)
    assert report["age_over_limit_count"] == 2
    assert report["age_limit_checked_step_count"] == 4
    assert report["age_limit_check_available"]


def test_old_camera_archive_does_not_invent_configured_limit(tmp_path):
    from dataclasses import replace

    path = tmp_path / "episode.npz"
    write_episode(path, steps=4)
    episode = replace(module.load_episode_archive(path), per_step_series={
        "camera_lag_from_step_end_s": np.full(4, 0.27),
        "camera_pair_reused": np.ones(4, dtype=bool),
    })
    report = module._camera_report(episode)
    assert report["age_over_limit_count"] == 0
    assert not report["age_limit_check_available"]
