from pathlib import Path

import numpy as np
import pytest

from so101_capstone.policies.so101_sim_calibration import (
    SIM_JOINT_CALIBRATIONS,
)
from so101_capstone.training.residual_ppo_gazebo import (
    FIRST_CHAIN_EPISODE_COUNT,
    FIRST_CHAIN_MAX_STEPS,
    FIRST_CHAIN_RESIDUAL_LIMIT_RAD,
    FIRST_CHAIN_ROLLOUT_STEPS,
    FIRST_CHAIN_TARGET_DELAY_S,
    ResidualPpoGazeboConfig,
    _default_output_directory,
    _delete_result_payload,
    _prepare_clean_episode,
    compose_residual_controller_command,
    validate_first_chain_contract,
)


def approved_config(**overrides):
    values = {
        "act_checkpoint": Path("checkpoint"),
        "device": "cuda",
        "episode_count": FIRST_CHAIN_EPISODE_COUNT,
        "max_steps_per_episode": FIRST_CHAIN_MAX_STEPS,
        "target_delay_s": FIRST_CHAIN_TARGET_DELAY_S,
        "rollout_steps": FIRST_CHAIN_ROLLOUT_STEPS,
        "residual_limit_rad": FIRST_CHAIN_RESIDUAL_LIMIT_RAD,
        "disable_max_joint_delta_limit": True,
    }
    values.update(overrides)
    return ResidualPpoGazeboConfig(**values)


def test_approved_first_chain_is_exactly_six_by_six_hundred():
    config = approved_config()
    validate_first_chain_contract(config)

    assert config.target_delay_steps == 90
    assert config.planned_transition_count == 3600
    assert config.to_json_dict()["max_joint_delta_rad"] is None


def test_first_chain_rejects_silent_parameter_changes():
    with pytest.raises(ValueError, match="fixed"):
        validate_first_chain_contract(
            approved_config(residual_limit_rad=0.05)
        )


def test_disable_max_joint_delta_flag_is_required():
    with pytest.raises(ValueError, match="fixed"):
        validate_first_chain_contract(
            approved_config(disable_max_joint_delta_limit=False)
        )


def test_composition_has_no_relative_step_delta_limit():
    base = np.zeros(6)
    residual = np.full(6, 0.1)
    command = compose_residual_controller_command(base, residual)
    np.testing.assert_allclose(command, residual)
    assert np.all(np.abs(command - base) > 0.03)


def test_composition_preserves_absolute_joint_limits():
    upper = np.asarray(
        [calibration.gazebo_upper_rad for calibration in SIM_JOINT_CALIBRATIONS]
    )
    lower = np.asarray(
        [calibration.gazebo_lower_rad for calibration in SIM_JOINT_CALIBRATIONS]
    )
    high = compose_residual_controller_command(upper, np.full(6, 0.1))
    low = compose_residual_controller_command(lower, np.full(6, -0.1))

    np.testing.assert_allclose(high, upper)
    np.testing.assert_allclose(low, lower)


def test_residual_larger_than_approved_bound_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        compose_residual_controller_command(
            np.zeros(6),
            np.array([0.1001, 0, 0, 0, 0, 0]),
        )


def test_non_integer_target_delay_is_rejected():
    with pytest.raises(ValueError, match="integer number"):
        approved_config(target_delay_s=3.01)



def test_default_output_directory_records_pretrained_lineage():
    checkpoint = Path(
        "output_simulation/pretrained/"
        "act_so101_pick_test_20260803_143945/"
        "training_output/checkpoints/060000/pretrained_model"
    )

    output = _default_output_directory(checkpoint)

    assert output.parent == Path("output_simulation/finetuned")
    assert output.name.endswith(
        "_from_pretrained_20260803_143945"
    )

def test_prepare_clean_episode_deletes_before_prepare():
    class Response:
        success = True
        message = "deleted"

    class Runtime:
        def __init__(self):
            self.events = []

        def delete_all_targets(self):
            self.events.append("delete_all")
            return Response()

        def prepare_episode(self):
            self.events.append("prepare_episode")
            return "prepared"

    runtime = Runtime()
    prepared, cleanup = _prepare_clean_episode(runtime, episode_index=3)

    assert prepared == "prepared"
    assert runtime.events == ["delete_all", "prepare_episode"]
    assert cleanup == {
        "requested": True,
        "stage": "before episode 3",
        "service_success": True,
        "service_message": "deleted",
    }


def test_delete_cleanup_rejects_service_failure():
    class Response:
        success = False
        message = "delete failed"

    with pytest.raises(RuntimeError, match="delete failed"):
        _delete_result_payload(Response(), stage="before episode 0")
