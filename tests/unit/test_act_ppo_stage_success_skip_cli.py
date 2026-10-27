"""Stage-success controls and logging without launching a simulator."""
import os
from pathlib import Path
import subprocess

import pytest

from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from test_act_ppo_tensorboard import FakeWriter, _logger
from test_act_ppo_stage_wave_integration import stage_config


def test_python_defaults_preserve_old_collection_contract():
    worlds = multi.MultiWorldConfig()
    args = multi._parser().parse_args(["--act-checkpoint", "/tmp/model"])
    assert worlds.stage_success_skip_enabled is False
    assert args.stage_success_skip is False
    assert args.stage_success_skip_threshold == .8


def test_cli_forwards_configurable_gate_to_main(monkeypatch, capsys):
    captured = []
    class Summary(dict):
        def __missing__(self, key):
            return "test"
    monkeypatch.setattr(multi, "run_act_ppo_multiworld_gazebo",
                        lambda base, worlds: captured.append(worlds) or Summary())
    monkeypatch.setattr("sys.argv", ["train", "--act-checkpoint", "/tmp/model",
                                    "--stage-success-skip", "--stage-success-skip-threshold", ".9"])
    multi.main()
    assert captured[0].stage_success_skip_enabled is True
    assert captured[0].stage_success_skip_threshold == .9


@pytest.mark.parametrize("invalid", [0, -1, 1.01, True, float("nan"), float("inf")])
def test_invalid_success_threshold_is_rejected_before_world_start(invalid):
    with pytest.raises(ValueError, match="stage_success_skip_threshold"):
        multi.MultiWorldConfig(stage_success_skip_threshold=invalid)


@pytest.mark.parametrize("filename", [None, "act_ppo_stages_dense.json"])
def test_legacy_stage_context_is_rejected_before_output_or_world_start(tmp_path, filename):
    path = None if filename is None else Path(__file__).resolve().parents[2] / "config" / filename
    base = stage_config(tmp_path, stage_config=path)
    worlds = multi.MultiWorldConfig(stage_success_skip_enabled=True)
    with pytest.raises(ValueError, match="requires process reward schema >= 2"):
        multi.run_act_ppo_multiworld_gazebo(base, worlds)
    assert not base.output_directory.exists()
    # Explicitly disabled preserves the legacy contract.
    assert multi._stage_success_skip_controller(base, multi.MultiWorldConfig()) is None


@pytest.mark.parametrize("enabled,threshold,stage,expected", [
    (None, None, "s0_s1", ["--stage-success-skip", "--stage-success-skip-threshold", "0.8"]),
    ("false", "0.9", "s0_s1", ["--no-stage-success-skip", "--stage-success-skip-threshold", "0.9"]),
    ("true", "0.85", "s1", ["--stage-success-skip", "--stage-success-skip-threshold", "0.85"]),
    (None, None, "full", []),
])
def test_shell_only_assembles_gate_arguments(enabled, threshold, stage, expected):
    script = (Path(__file__).resolve().parents[2] / "scripts/finetune_act_ppo_so101.sh").read_text()
    assignments = "\n".join(line for line in script.splitlines()
                            if line.startswith(("STAGE_SUCCESS_SKIP_ENABLED=", "STAGE_SUCCESS_SKIP_THRESHOLD=")))
    start = script.index("STAGE_SUCCESS_SKIP_ARGS=()")
    end = script.index('if [[ "$REWARD_PROFILE" == "pick_place_v2" ]]', start)
    # Evaluate only assignments and argument assembly, never the launcher.
    code = assignments + "\n" + script[start:end] + '\nprintf "\\0%s\\0" "${STAGE_SUCCESS_SKIP_ARGS[@]}"'
    env = dict(os.environ, TRAINING_STAGES=stage)
    for name, value in (("STAGE_SUCCESS_SKIP_ENABLED", enabled), ("STAGE_SUCCESS_SKIP_THRESHOLD", threshold)):
        env.pop(name, None)
        if value is not None:
            env[name] = value
    result = subprocess.run(["bash", "-c", code], env=env, capture_output=True, text=True, check=True)
    assert [item for item in result.stdout.split("\0")[1:] if item] == expected
    assert '"${STAGE_SUCCESS_SKIP_ARGS[@]}"' in script[script.index("python -m so101_capstone.training.act_ppo_multiworld_gazebo"):]


def test_skip_tensorboard_uses_collection_batch_index_not_ppo_index():
    writer = FakeWriter()
    _logger(writer).log_stage_success_skip({
        "batch_index": 7, "success_rates": {"S0": .8, "S1": None},
        "exclude_update_stages": ["S0"], "next_skip_stages": ["S0"],
        "collected_transition_count": 2400, "update_transition_count": 1500,
        "excluded_transition_count": 900, "empty_batch": False,
    })
    scalars = {(tag, step): value for tag, value, step in writer.scalars}
    assert scalars["stage_skip/S0/success_rate", 7] == .8
    assert ("stage_skip/S1/success_rate", 7) not in scalars
    assert scalars["stage_skip/excluded_transition_count", 7] == 900
    assert scalars["stage_skip/S0/next_skip_stages", 7] == 1
    assert scalars["stage_skip/S1/next_skip_stages", 7] == 0
