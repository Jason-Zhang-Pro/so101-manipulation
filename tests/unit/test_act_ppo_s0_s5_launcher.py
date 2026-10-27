"""Exercise the real launcher while replacing the trainer with a harmless shell stub."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/finetune_act_ppo_s0_s5.sh"
GOOD40 = (
    "output_simulation/finetuned/act_ppo_20261001_182324_from_checkpoints/"
    "checkpoints/update_000040/pretrained_model"
)


def write_model(path, weight_name="model.safetensors"):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"type": "act"}')
    (path / weight_name).write_bytes(b"stub weights; never loaded")
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(LAUNCHER, root / "scripts" / LAUNCHER.name)
    # The wrapper validates weights with the same dependency-free scheduler parser.
    schedule = Path("src/so101_capstone/training/act_ppo_stage_schedule.py")
    (root / schedule).parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / schedule, root / schedule)
    (root / "config").mkdir()
    for name in (
        "act_ppo_stages_s0_s5.json", "pick_place_reward.json",
        "environment_randomization.json", "image_augmentation_s0_s1.json",
    ):
        (root / "config" / name).write_text("{}")
    write_model(root / GOOD40)
    # Real training/conda/ROS/Gazebo are not invoked, including in dispatch tests.
    (root / "scripts/finetune_act_ppo_so101.sh").write_text(
        "#!/usr/bin/env bash\n"
        "printf 'TRAINER_STUB\\n'\n"
        "env -0\n"
    )
    return root


def run_launcher(project, *args, overrides=None):
    env = {"PATH": os.environ["PATH"], "HOME": str(project / "no_conda_here")}
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(project / "scripts" / LAUNCHER.name), *args],
        env=env, text=True, capture_output=True, check=False, timeout=10,
    )


def parse_config(result, inherited=None):
    assert result.returncode == 0, result.stderr
    assert "TRAINER_STUB" not in result.stdout
    result = subprocess.run(
        ["bash", "-c", "source /dev/stdin\nenv -0"],
        input=result.stdout, text=True, capture_output=True, check=True, timeout=10,
        env={"PATH": os.environ["PATH"], **(inherited or {})},
    )
    return dict(item.split("=", 1) for item in result.stdout.split("\0") if item)


def test_default_profile_uses_good40_effective_learning_rates_with_all_stages(project):
    config = parse_config(run_launcher(project, "--print-config"))
    assert config["PROJECT_ROOT"] == str(project)
    assert config["PRETRAINED_MODEL"] == str(project / GOOD40)
    assert config["STAGE_CONFIG"] == str(project / "config/act_ppo_stages_s0_s5.json")
    expected = {
        "TRAINING_STAGES": "s0_s5", "DEVICE": "cuda", "ACT_MAX_ACTION_STEPS": "1",
        "WORLD_COUNT": "4", "GAZEBO_GUI": "false", "EPISODE_COUNT": "100000000",
        "ROLLOUT_STEPS": "2400", "MINIBATCH_SIZE": "30", "PPO_EPOCHS": "2",
        "STAGE_STEP_WEIGHTS": "1:3:2:2:1:1",
        "TARGET_KL": "50", "KL_BACKOFF_MAX_RETRIES": "10",
        "KL_BACKOFF_FACTOR": "0.5", "KL_BACKOFF_MIN_SCALE": "0.0064",
        "ACTOR_LEARNING_RATE": "3.125e-6", "ACTOR_BACKBONE_LEARNING_RATE": "3.125e-7",
        "CRITIC_LEARNING_RATE": "2e-4", "EXPLORATION_LEARNING_RATE": "3.125e-6",
        "RAW_MEAN_BOUNDS_COEFFICIENT": "0.1", "CHECKPOINT_EVERY_UPDATES": "5",
        "EXPLORATION_PROFILE": "final_good_act_near_target",
        "STAGE_SUCCESS_SKIP_ENABLED": "true", "STAGE_SUCCESS_SKIP_THRESHOLD": "0.8",
        "S0_BUDGET_S": "5.0", "S0_HOME_HOLD_S": "1.0", "S1_REACH_HOLD_S": "0",
        "S1_BUDGET_S": "10.0", "S2_BUDGET_S": "8.0", "S3_BUDGET_S": "8.0",
        "S4_BUDGET_S": "8.0", "STAGE_INITIALIZATION_TIMEOUT_S": "600.0",
        "IMAGE_AUGMENTATION": "true", "ENVIRONMENT_RANDOMIZATION": "false",
        "BASE_ROS_DOMAIN_ID": "20", "GZ_PARTITION_PREFIX": "gz_env_",
        "TARGET_DELAY_S": "0", "FINAL_TARGET_ABSENT_S": "0",
    }
    assert {key: config[key] for key in expected} == expected
    assert "S5_BUDGET_S" not in config  # S5 reads the same S0 budget in Python.
    assert not (project / "output_simulation/finetuned").joinpath("new_run").exists()
    assert not (project / "no_conda_here").exists()


def test_print_and_real_dispatch_both_clear_stale_legacy_and_resume_values(project):
    stale = {
        "RESUME_CHECKPOINT": "/old/train_state.pt", "S0_PROBABILITY": "0.99",
        "STATE1_ACTOR_UPDATE_ENABLED": "false", "STATE3_EXPLORATION_LOG_STD": "0",
        "IDLE_EXPLORATION_LOG_STD": "0", "TRAINING_STAGES": "s0_s1",
        "STAGE_CONFIG": "/wrong/stages.json", "PROJECT_ROOT": "/wrong/project",
        "TARGET_KL": "0.02", "BASE_ROS_DOMAIN_ID": "99",
        "STAGE_SUCCESS_SKIP_ENABLED": "false", "RAW_MEAN_BOUNDS_COEFFICIENT": "0",
    }
    result = run_launcher(project, "--print-config", overrides=stale)
    printed = parse_config(result, inherited=stale)
    actual = run_launcher(project, overrides=stale)
    assert actual.returncode == 0, actual.stderr
    assert actual.stdout.startswith("TRAINER_STUB\n")
    dispatched = dict(
        item.split("=", 1) for item in actual.stdout.removeprefix("TRAINER_STUB\n").split("\0") if item
    )
    names = re.findall(r"^export (\w+)=", result.stdout, re.MULTILINE)
    assert {name: dispatched[name] for name in names} == {name: printed[name] for name in names}
    for name in (
        "RESUME_CHECKPOINT", "S0_PROBABILITY", "STATE1_ACTOR_UPDATE_ENABLED",
        "STATE3_EXPLORATION_LOG_STD", "IDLE_EXPLORATION_LOG_STD",
    ):
        assert name not in printed and name not in dispatched
    assert dispatched["BASE_ROS_DOMAIN_ID"] == "20"
    assert dispatched["STAGE_SUCCESS_SKIP_ENABLED"] == "true"


def test_explicit_overrides_and_checkpoint_shell_metacharacters_are_preserved(project):
    model = write_model(project / "models" / "literal ' $(false) `false` $PATH weights")
    overrides = {
        "PRETRAINED_MODEL": str(model), "WORLD_COUNT": "6", "GAZEBO_GUI": "true",
        "EPISODE_COUNT": "24", "STAGE_STEP_WEIGHTS": "2:4:3:2:1:2",
    }
    config = parse_config(run_launcher(project, "--print-config", overrides=overrides))
    assert {name: config[name] for name in overrides} == overrides


@pytest.mark.parametrize("args", [("--unknown",), ("--print-config", "extra"), ("--execute",)])
def test_unknown_arguments_never_launch(project, args):
    result = run_launcher(project, *args)
    assert result.returncode == 2 and "Unknown arguments" in result.stderr
    assert "TRAINER_STUB" not in result.stdout


@pytest.mark.parametrize("overrides", [
    {"WORLD_COUNT": "0"}, {"WORLD_COUNT": "1.5"}, {"EPISODE_COUNT": "-1"},
    {"GAZEBO_GUI": "yes"},
])
def test_invalid_overrides_fail_before_dispatch(project, overrides):
    result = run_launcher(project, overrides=overrides)
    assert result.returncode == 2 and "Error:" in result.stderr
    assert "TRAINER_STUB" not in result.stdout


@pytest.mark.parametrize("weights", ["", "1:2:3", "1:3:2:2:1:1:", "0:3:2:2:1:1",
                                    "1:-3:2:2:1:1", "1:nan:2:2:1:1", "1:1e999:2:2:1:1"])
@pytest.mark.parametrize("args", [(), ("--print-config",)])
def test_invalid_stage_weights_fail_before_print_or_dispatch(project, weights, args):
    result = run_launcher(project, *args, overrides={"STAGE_STEP_WEIGHTS": weights})
    assert result.returncode == 2 and "STAGE_STEP_WEIGHTS" in result.stderr
    assert "TRAINER_STUB" not in result.stdout


def test_weight_override_survives_actual_dispatch(project):
    result = run_launcher(project, overrides={"STAGE_STEP_WEIGHTS": "1:4:2:1:1:1"})
    assert result.returncode == 0 and result.stdout.startswith("TRAINER_STUB\n")
    values = dict(item.split("=", 1) for item in result.stdout.removeprefix("TRAINER_STUB\n").split("\0") if item)
    assert values["STAGE_STEP_WEIGHTS"] == "1:4:2:1:1:1"


@pytest.mark.parametrize("file", [
    "config/act_ppo_stages_s0_s5.json", "config/pick_place_reward.json",
    GOOD40 + "/config.json", GOOD40 + "/model.safetensors",
])
def test_missing_model_or_configuration_fails_before_dispatch(project, file):
    (project / file).unlink()
    result = run_launcher(project)
    assert result.returncode == 2 and "Error:" in result.stderr
    assert "TRAINER_STUB" not in result.stdout


def test_pytorch_bin_checkpoint_is_supported(project):
    model = write_model(project / "pytorch model", "pytorch_model.bin")
    result = run_launcher(project, "--print-config", overrides={"PRETRAINED_MODEL": str(model)})
    assert parse_config(result)["PRETRAINED_MODEL"] == str(model)


@pytest.mark.parametrize("argument", ["--help", "-h"])
def test_help_does_not_require_checkpoint_or_start_training(project, argument):
    (project / GOOD40 / "config.json").unlink()
    result = run_launcher(project, argument)
    assert result.returncode == 0 and "Usage:" in result.stdout
    assert "TRAINER_STUB" not in result.stdout
