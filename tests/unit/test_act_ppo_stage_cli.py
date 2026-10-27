"""Stage time overrides reach the same runtime contract in both launch paths."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training import act_ppo_gazebo as single
from so101_capstone.training import act_ppo_multiworld_gazebo as multi


TIME_FIELDS = ("s0_budget_s", "s0_home_hold_s", "s1_budget_s", "s1_reach_hold_s")


def build_config(arguments, *, multiworld=False):
    parser = multi._parser() if multiworld else single._parser()
    args = parser.parse_args(["--act-checkpoint", "/tmp/actor", *arguments])
    if multiworld:
        return multi._build_base_config(args)
    return single.ActPpoGazeboConfig(
        act_checkpoint=args.act_checkpoint, final_target_absent_s=args.final_target_absent_s,
        **single._reward_cli_kwargs(args),
    )


@pytest.mark.parametrize("multiworld", [False, True])
def test_stage_times_override_json_and_match_effective_checkpoint_contract(tmp_path, multiworld):
    path = tmp_path / "stages with spaces.json"
    base = replace(StageTrainingConfig(), s0_budget_s=7, s1_reach_xy_m=.009)
    path.write_text(json.dumps(base.to_dict()))
    arguments = ["--training-stages", "s0_s1", "--stage-config", str(path)]
    inherited = build_config(arguments, multiworld=multiworld)
    assert all(getattr(inherited, name) is None for name in TIME_FIELDS)
    assert single._resolved_stage_config(inherited) == base

    overridden = build_config(arguments + [
        "--s0-budget-s", "2", "--s0-home-hold-s", "1",
        "--s1-budget-s", "15", "--s1-reach-hold-s", "0.3",
    ], multiworld=multiworld)
    expected = replace(base, s0_budget_s=2, s0_home_hold_s=1, s1_budget_s=15, s1_reach_hold_s=.3)
    assert single._resolved_stage_config(overridden) == expected
    assert json.loads(path.read_text()) == base.to_dict()
    assert single._episode_step_limit(overridden, "idle_empty") == 61
    assert single._episode_step_limit(overridden, "pick_place") == 451
    assert overridden.planned_transition_count == overridden.episode_count * 450
    assert overridden.to_json_dict()["s0_budget_s"] == 2
    signature = single._execution_signature(overridden, single._resolved_reward(overridden))
    assert signature["stage_training"]["config"] == expected.to_dict()
    assert signature != single._execution_signature(inherited, single._resolved_reward(inherited))

    path.write_text(json.dumps(expected.to_dict()))
    from_json = build_config(arguments, multiworld=multiworld)
    assert signature == single._execution_signature(from_json, single._resolved_reward(from_json))


@pytest.mark.parametrize("multiworld", [False, True])
def test_instant_distance_profile_accepts_zero_hold_and_rejects_stale_override(multiworld):
    path = Path(__file__).resolve().parents[2] / "config/act_ppo_stages_s1_posture_guard.json"
    arguments = ["--training-stages", "s0_s1", "--stage-config", str(path)]
    inherited = build_config(arguments, multiworld=multiworld)
    explicit = build_config(arguments + ["--s1-reach-hold-s", "0"], multiworld=multiworld)
    resolved = single._resolved_stage_config(inherited)
    assert resolved.s1_success_mode == "lift_then_distance"
    assert resolved.s1_reach_hold_s == 0
    assert single._resolved_stage_config(explicit) == resolved
    assert single._execution_signature(inherited, single._resolved_reward(inherited)) == single._execution_signature(
        explicit, single._resolved_reward(explicit))
    with pytest.raises(ValueError, match="s1_reach_hold_s=0"):
        build_config(arguments + ["--s1-reach-hold-s", "0.15"], multiworld=multiworld)


@pytest.mark.parametrize("name", TIME_FIELDS)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_duration_rejected_before_runtime(name, value):
    with pytest.raises(ValueError, match="positive|finite"):
        build_config(["--training-stages", "s0_s1", "--" + name.replace("_", "-"), value])


@pytest.mark.parametrize("arguments", [
    ["--s0-budget-s", "2"], ["--s0-home-hold-s", "6"],
    ["--s1-budget-s", "0.1"], ["--s1-reach-hold-s", "11"],
])
def test_hold_must_fit_merged_budget(arguments):
    with pytest.raises(ValueError, match="hold must fit"):
        build_config(["--training-stages", "s0_s1", *arguments])


@pytest.mark.parametrize("name", TIME_FIELDS)
def test_full_task_does_not_silently_ignore_stage_time(name):
    with pytest.raises(ValueError, match="require S0/S1"):
        build_config(["--" + name.replace("_", "-"), "1"])


@pytest.mark.parametrize("value", [None, "", "12", "0"])
def test_launch_script_only_forwards_nonempty_time_overrides(value):
    script = (Path(__file__).resolve().parents[2] / "scripts" / "finetune_act_ppo_so101.sh").read_text()
    # Execute only the real variable defaults and argument assembly; never
    # source the launch script or start its simulation/preflight processes.
    names = ("TRAINING_STAGES", "STAGE_CONFIG", "S0_PROBABILITY", "REWARD_PROFILE", "TASK_MODE",
             "ACT_MAX_ACTION_STEPS", "REWARD_CONFIG", *(name.upper() for name in TIME_FIELDS))
    assignments = "\n".join(line for line in script.splitlines()
                            if any(line.startswith(name + "=") for name in names))
    assembly = script[script.index("REWARD_ARGS=("):script.index("EXPLORATION_ARGS=()")]
    env = {key: val for key, val in os.environ.items() if key not in names}
    if value is not None:
        env.update({name.upper(): value for name in TIME_FIELDS})
    result = subprocess.run(
        ["bash", "-c", assignments + "\n" + assembly + '\nprintf "%s\\0" "${REWARD_ARGS[@]}"'],
        env=env, check=True, capture_output=True,
    )
    arguments = result.stdout.decode().rstrip("\0").split("\0")
    args = single._parser().parse_args(["--act-checkpoint", "/tmp/actor", *arguments])
    for name in TIME_FIELDS:
        assert getattr(args, name) == (None if value in (None, "") else float(value))
    if value == "0":
        with pytest.raises(ValueError, match="positive"):
            multi._build_base_config(multi._parser().parse_args(["--act-checkpoint", "/tmp/actor", *arguments]))
