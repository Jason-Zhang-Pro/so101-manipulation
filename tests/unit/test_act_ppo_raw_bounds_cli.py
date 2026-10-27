"""Raw ACT output regularization reaches both launchers without a simulator."""

from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys

import pytest

from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi
from test_act_ppo_tensorboard import FakeWriter, _logger


@pytest.mark.parametrize("entry", ["single", "multiworld"])
@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.25])
def test_raw_bounds_cli_reaches_the_ppo_objective(monkeypatch, entry, coefficient):
    arguments = ["--act-checkpoint", "/tmp/model", "--raw-mean-bounds-coefficient", str(coefficient)]
    if entry == "single":
        captured = []
        monkeypatch.setattr(sys, "argv", ["act-ppo"] + arguments)
        monkeypatch.setattr(single, "run_act_ppo_gazebo",
                            lambda config: captured.append(config) or defaultdict(lambda: "test"))
        single.main()
        base = captured[0]
    else:
        base = multi._build_base_config(multi._parser().parse_args(arguments))
    assert single._act_ppo_config(base).raw_mean_bounds_coefficient == coefficient
    assert base.to_json_dict()["raw_mean_bounds_coefficient"] == coefficient


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True])
def test_invalid_raw_bounds_coefficient_fails_before_simulation(invalid):
    with pytest.raises(ValueError, match="raw_mean_bounds_coefficient"):
        single.ActPpoGazeboConfig(act_checkpoint=Path("/tmp/model"), raw_mean_bounds_coefficient=invalid)


def test_launcher_enables_regularizer_and_allows_explicit_zero():
    script = Path(__file__).resolve().parents[2] / "scripts/finetune_act_ppo_so101.sh"
    name = "RAW_MEAN_BOUNDS_COEFFICIENT"
    assignment = next(line for line in script.read_text().splitlines() if line.startswith(name + "="))
    for value, expected in [(None, "0.1"), ("0", "0"), ("0.25", "0.25")]:
        env = dict(os.environ)
        env.pop(name, None)
        if value is not None:
            env[name] = value
        # Execute only the isolated assignment, never the training launcher.
        result = subprocess.run(["bash", "-c", assignment + '\nprintf "%s" "$' + name + '"'],
                                env=env, capture_output=True, text=True, check=True)
        assert result.stdout == expected


@pytest.mark.parametrize("count", [0, 24])
def test_raw_tensorboard_diagnostics_distinguish_disabled_from_zero_overflow(count):
    writer = FakeWriter()
    _logger(writer).log_update({
        "update_index": 3,
        "raw_mean_bounds_diagnostic_count": count,
        "raw_mean_bounds_loss": 0.02,
        "raw_mean_out_of_bounds_fraction": 0.5,
        "raw_mean_overflow_normalized": 0.1,
        "raw_mean_bound_derivative_mean": 0.25,
    }, total_environment_steps=123)
    scalars = {(tag, step): value for tag, value, step in writer.scalars}
    assert scalars[("optimization/raw_bounds_diagnostic_count", 3)] == count
    expected = {
        "loss/raw_mean_bounds_loss": 0.02,
        "action/raw_mean_out_of_bounds_fraction": 0.5,
        "action/raw_mean_overflow_normalized": 0.1,
        "action/raw_mean_bound_derivative_mean": 0.25,
    }
    for tag, value in expected.items():
        if count:
            assert scalars[(tag, 3)] == value
        else:
            assert (tag, 3) not in scalars


def test_nonfinite_raw_diagnostics_are_not_plotted():
    writer = FakeWriter()
    _logger(writer).log_update({
        "update_index": 1, "raw_mean_bounds_diagnostic_count": 1,
        "raw_mean_bounds_loss": float("nan"),
        "raw_mean_out_of_bounds_fraction": float("inf"),
        "raw_mean_overflow_normalized": float("nan"),
        "raw_mean_bound_derivative_mean": float("nan"),
    }, total_environment_steps=1)
    tags = {tag for tag, _, _ in writer.scalars}
    assert not any(tag.startswith("action/raw_mean_") or tag == "loss/raw_mean_bounds_loss" for tag in tags)
