"""KL retry controls must reach both training entry points without simulation."""

from pathlib import Path
import os
import subprocess

import pytest

from so101_capstone.training.act_ppo_gazebo import ActPpoGazeboConfig, _act_ppo_config


@pytest.mark.parametrize("entry", ["single", "multiworld"])
def test_kl_backoff_cli_reaches_trainer_config(monkeypatch, entry):
    from collections import defaultdict
    import sys
    from so101_capstone.training import act_ppo_gazebo as single
    from so101_capstone.training import act_ppo_multiworld_gazebo as multi

    arguments = ["--act-checkpoint", "/tmp/pretrained_model", "--target-kl", "0.03",
                 "--kl-backoff-max-retries", "4", "--kl-backoff-factor", "0.4",
                 "--kl-backoff-min-scale", "0.002"]
    if entry == "single":
        captured = []
        monkeypatch.setattr(sys, "argv", ["act-ppo"] + arguments)
        monkeypatch.setattr(single, "run_act_ppo_gazebo",
                            lambda config: captured.append(config) or defaultdict(lambda: "test"))
        single.main()
        base = captured[0]
    else:
        base = multi._build_base_config(multi._parser().parse_args(arguments))
    resolved = _act_ppo_config(base)
    assert resolved.target_kl == 0.03
    assert resolved.kl_backoff_max_retries == 4
    assert resolved.kl_backoff_factor == 0.4
    assert resolved.kl_backoff_min_scale == 0.002


@pytest.mark.parametrize("field,value", [
    ("kl_backoff_max_retries", -1), ("kl_backoff_max_retries", True),
    ("kl_backoff_max_retries", 1.5), ("kl_backoff_factor", 0),
    ("kl_backoff_factor", 1), ("kl_backoff_factor", float("nan")),
    ("kl_backoff_factor", True), ("kl_backoff_min_scale", 0),
    ("kl_backoff_min_scale", 1.1), ("kl_backoff_min_scale", float("inf")),
    ("kl_backoff_min_scale", True),
])
def test_invalid_backoff_controls_fail_before_starting_world(field, value):
    with pytest.raises(ValueError, match=field):
        ActPpoGazeboConfig(act_checkpoint=Path("/tmp/model"), **{field: value})


@pytest.mark.parametrize("name,default,override", [
    ("KL_BACKOFF_MAX_RETRIES", "8", "0"),
    ("KL_BACKOFF_FACTOR", "0.5", "0.25"),
    ("KL_BACKOFF_MIN_SCALE", "1e-4", "0.001"),
])
def test_launcher_enables_bounded_backoff_and_preserves_overrides(name, default, override):
    script = Path(__file__).resolve().parents[2] / "scripts/finetune_act_ppo_so101.sh"
    assignment = next(line for line in script.read_text().splitlines() if line.startswith(name + "="))
    for provided, expected in ((None, default), (override, override)):
        environment = dict(os.environ)
        environment.pop(name, None)
        if provided is not None:
            environment[name] = provided
        result = subprocess.run(["bash", "-c", assignment + '\nprintf "%s" "$' + name + '"'],
                                env=environment, capture_output=True, text=True, check=True)
        assert result.stdout == expected
