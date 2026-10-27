"""Execute only launch-script defaults and the stage argument array."""

import os
from pathlib import Path
import re
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "finetune_act_ppo_so101.sh"
SCHEDULE_VARIABLES = (
    "S0_STEP_FRACTION", "STAGE_RATIO_TOLERANCE", "STAGE_MAX_ROLLOUT_MULTIPLIER",
)


def assembled_schedule(overrides):
    script = SCRIPT.read_text()
    # This prefix contains only stage variable definitions and the legacy
    # parameter notice. Never source ROS, run cleanup, or launch training.
    prefix = script[script.index('REWARD_PROFILE="'):script.index("# 仿真秒")]
    match = re.search(r"STAGE_SCHEDULE_ARGS=\([\s\S]*?\)\n", script)
    assert match is not None
    variables = (*SCHEDULE_VARIABLES, "TRAINING_STAGES", "STAGE_CONFIG", "S0_PROBABILITY")
    env = {name: value for name, value in os.environ.items() if name not in variables}
    env.update(overrides)
    result = subprocess.run(
        ["bash", "-c", prefix + "\n" + match.group() + '\nprintf "%s\\0" "${STAGE_SCHEDULE_ARGS[@]}"'],
        env=env, check=True, capture_output=True,
    )
    arguments = result.stdout.decode().rstrip("\0").split("\0")
    return dict(zip(arguments[::2], arguments[1::2])), result.stderr.decode()


@pytest.mark.parametrize("overrides", [{}, {name: "" for name in SCHEDULE_VARIABLES}])
def test_stage_schedule_defaults_and_empty_values(overrides):
    arguments, notice = assembled_schedule(overrides)
    assert arguments == {
        "--s0-step-fraction": "0.4",
        "--stage-ratio-tolerance": "0.05",
        "--stage-max-rollout-multiplier": "4",
    }
    assert notice == ""


def test_custom_stage_schedule_values_reach_separate_multiworld_arguments():
    arguments, _ = assembled_schedule({
        "S0_STEP_FRACTION": "0.35", "STAGE_RATIO_TOLERANCE": "0.01",
        "STAGE_MAX_ROLLOUT_MULTIPLIER": "7",
    })
    assert arguments == {
        "--s0-step-fraction": "0.35",
        "--stage-ratio-tolerance": "0.01",
        "--stage-max-rollout-multiplier": "7",
    }
    script = SCRIPT.read_text()
    launch = script[script.index("python -m so101_capstone.training.act_ppo_multiworld_gazebo"):]
    assert '"${STAGE_SCHEDULE_ARGS[@]}"' in launch


def test_legacy_episode_probability_is_never_reinterpreted_as_step_fraction():
    arguments, notice = assembled_schedule({"S0_PROBABILITY": "0.625"})
    assert arguments["--s0-step-fraction"] == "0.4"
    assert "不再使用S0_PROBABILITY" in notice
    assert "S0_STEP_FRACTION" in notice


@pytest.mark.parametrize("s0_seconds,s1_seconds", [("5", "10"), ("6", "15"), ("10", "3")])
def test_step_fraction_does_not_depend_on_budget_ratio(s0_seconds, s1_seconds):
    arguments, _ = assembled_schedule({
        "S0_BUDGET_S": s0_seconds, "S1_BUDGET_S": s1_seconds,
        "S0_STEP_FRACTION": "0.4",
    })
    assert arguments["--s0-step-fraction"] == "0.4"
