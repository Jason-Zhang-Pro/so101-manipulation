"""Check the S1-only near-target noise change through sampling and resume.

These are offline distribution checks, not evidence of better task success.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from so101_capstone.policies.so101_sim_calibration import policy_positions_to_gazebo
from so101_capstone.rewards.training_stages import StageTrainingConfig
from so101_capstone.training.act_ppo import (
    ActPpoExploration,
    ActPpoTrainer,
    ActPpoValueNetwork,
    evaluate_bounded_policy_actions,
    sample_bounded_policy_actions,
)
from so101_capstone.training.act_ppo_exploration_profile import (
    final_good_act_exploration_signature,
    validate_final_good_act_exploration_bounds,
)
from so101_capstone.training.act_ppo_gazebo import (
    _act_ppo_config,
    _execution_signature,
    _exploration_stage,
)
from so101_capstone.training.residual_ppo import model_state_sha256
from test_act_ppo import TinyActPolicy
from test_act_ppo_distance_exploration import HALF_SPANS, _exploration, _offset, _rad_stds
from test_stage_process_integration import PROJECT, REWARD, small_ppo_config, training_config


@pytest.fixture
def launch_config(tmp_path):
    stages = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_s1_progress.json")
    return replace(
        training_config(tmp_path, stages, "near_exploration"),
        exploration_profile="final_good_act", initial_log_std=-3.,
        minimum_log_std=-9., maximum_log_std=-2.5, gripper_maximum_log_std=-1.5,
    )


@pytest.mark.parametrize("distance,expected", [
    (0., .12), (.019, .12), (.02, .12), (.075, .12), (.08, .12),
    (np.nextafter(.08, np.inf), .08), (.1, .08), (.15, .08),
    (np.nextafter(.15, np.inf), .12),
])
def test_s1_approach_boundary_and_real_effective_std(distance, expected):
    np.testing.assert_allclose(_rad_stds({
        "training_stage": "S1", "task_phase": "APPROACH",
        "stage_reach_distance_m": distance,
    }), [expected] * 6, rtol=4e-7)


@pytest.mark.parametrize("stage", [None, "S0", "S2", "full"])
def test_near_approach_without_s1_preserves_old_profile(stage):
    info = {"task_phase": "APPROACH", "target_distance_m": .03}
    if stage is not None:
        info["training_stage"] = stage
    np.testing.assert_allclose(_rad_stds(info, "manipulation" if stage == "S2" else "moving"),
                               [.05] * 6, rtol=4e-7)


@pytest.mark.parametrize("phase,group", [
    ("GRASP_LIFT", "manipulation"), ("TRANSPORT", "moving"),
    ("PLACE_RELEASE", "manipulation"),
])
@pytest.mark.parametrize("stage,expected", [("S1", .12), ("S0", .05), (None, .05)])
def test_s1_close_target_phases_keep_large_noise_across_grasp_lift_boundary(phase, group, stage, expected):
    info = {"task_phase": phase, "target_distance_m": .03}
    if stage is not None:
        info["training_stage"] = stage
    np.testing.assert_allclose(_rad_stds(info, group), [expected] * 6, rtol=4e-7)


@pytest.mark.parametrize("phase,group", [
    ("WAIT_HOME", "idle"), ("RETURN_HOME", "moving"), ("HOME_HOLD", "home_hold"),
])
@pytest.mark.parametrize("stage", ["S0", "S1"])
@pytest.mark.parametrize("error,expected", [(.02, .001), (.08, .031)])
def test_home_taper_unchanged_even_with_near_target(stage, phase, group, error, expected):
    np.testing.assert_allclose(_rad_stds({
        "training_stage": stage, "task_phase": phase,
        "stage_home_error_rad": error, "home_joint_tolerance_rad": .02,
        "stage_reach_distance_m": .03,
    }, group), [expected] * 6, rtol=5e-7)


def test_override_uses_stage_distance_with_full_task_fallback():
    info = {"training_stage": "S1", "task_phase": "APPROACH", "target_distance_m": .03}
    np.testing.assert_allclose(_rad_stds(info), [.12] * 6, rtol=4e-7)
    # A present stage measurement takes precedence over the fallback, in both directions.
    np.testing.assert_allclose(_rad_stds({**info, "stage_reach_distance_m": .1}),
                               [.08] * 6, rtol=4e-7)
    np.testing.assert_allclose(_rad_stds({**info, "target_distance_m": .1,
                                        "stage_reach_distance_m": .03}),
                               [.12] * 6, rtol=4e-7)
    with pytest.raises(ValueError, match="stage_reach_distance_m"):
        _offset({**info, "stage_reach_distance_m": float("nan")})


def test_real_collector_selector_and_ppo_bounds_keep_requested_physical_std(launch_config):
    info = {"training_stage": "S1", "task_phase": "APPROACH", "stage_reach_distance_m": .04}
    group, _, offset = _exploration_stage(info, launch_config)
    assert group == "moving"
    ppo = _act_ppo_config(launch_config, 1)
    logs = ActPpoExploration(ppo).with_offset(offset).detach().double()
    np.testing.assert_allclose(logs.exp().numpy() * HALF_SPANS, [.12] * 6, rtol=4e-7)
    assert torch.all(logs[:5] <= ppo.maximum_log_std)
    assert logs[-1] <= ppo.gripper_maximum_log_std
    validate_final_good_act_exploration_bounds(
        initial_log_std=ppo.initial_log_std, minimum_log_std=ppo.minimum_log_std,
        maximum_log_std=ppo.maximum_log_std, gripper_maximum_log_std=ppo.gripper_maximum_log_std,
    )


def test_same_seed_near_commands_are_2_4_times_larger_bidirectional_and_replay_exactly():
    means = torch.tensor([0.] * 5 + [50.], dtype=torch.float64).expand(4096, 6)
    centre = policy_positions_to_gazebo(means[0].numpy())
    rms = []
    exploration = _exploration()
    old_info = {"task_phase": "APPROACH", "target_distance_m": .04}
    for info, sigma in ((old_info, .05), ({**old_info, "training_stage": "S1"}, .12)):
        logs = exploration.with_offset(_offset(info))
        torch.manual_seed(732)
        actions, recorded_log_prob, latent = sample_bounded_policy_actions(
            means, logs, return_latent=True,
        )
        replay, _ = evaluate_bounded_policy_actions(
            means, actions, logs, policy_action_latents=latent,
        )
        torch.testing.assert_close(replay, recorded_log_prob, rtol=0., atol=0.)
        assert bool((latent.abs() < 1.).all()), "Use unclipped samples for the physical RMS comparison."
        commands = np.stack([policy_positions_to_gazebo(row) for row in actions.detach().numpy()])
        deviations = commands - centre
        assert np.all((deviations < 0.).any(axis=0))
        assert np.all((deviations > 0.).any(axis=0))
        rms.append(np.sqrt(np.mean(deviations ** 2, axis=0)))
        np.testing.assert_allclose(rms[-1], [sigma] * 6, rtol=.04)
    np.testing.assert_allclose(rms[1] / rms[0], [2.4] * 6, rtol=1e-6)


def test_signature_records_s1_scope_and_is_not_shared_mutable_state():
    signature = final_good_act_exploration_signature()
    assert signature["schema_version"] == 2
    assert signature["s1_near_override"] == {
        "training_stage": "S1", "phases": ["APPROACH", "GRASP_LIFT", "PLACE_RELEASE", "TRANSPORT"],
        "distance_upper_inclusive_m": .08, "initial_joint_std_rad": .12,
    }
    assert signature["distance_bins"]["initial_joint_std_rad"] == [.05, .08, .12]
    signature["s1_near_override"]["phases"].append("HOME_HOLD")
    assert final_good_act_exploration_signature()["s1_near_override"]["phases"] == [
        "APPROACH", "GRASP_LIFT", "PLACE_RELEASE", "TRANSPORT",
    ]


@pytest.mark.parametrize("old_is_source", [True, False])
def test_changed_exploration_cannot_silently_resume_previous_contract(
    launch_config, tmp_path, old_is_source,
):
    current = _execution_signature(launch_config, REWARD)
    old = deepcopy(current)
    old["exploration_profile"]["schema_version"] = 1
    old["exploration_profile"].pop("s1_near_override")
    source, target = (old, current) if old_is_source else (current, old)
    ppo = small_ppo_config(launch_config)

    def trainer(signature):
        return ActPpoTrainer(TinyActPolicy(chunk_size=1), ActPpoValueNetwork(ppo),
                            config=ppo, execution_signature=signature)

    original = trainer(source)
    path = original.save_checkpoint(tmp_path / "training_state.pt")
    matching = trainer(source)
    matching.load_checkpoint(path)
    assert matching.execution_signature_sha256 == original.execution_signature_sha256
    assert model_state_sha256(matching.policy) == model_state_sha256(original.policy)
    incompatible = trainer(target)
    modules = (incompatible.policy, incompatible.critic, incompatible.exploration)
    before = tuple(model_state_sha256(module) for module in modules)
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path)
    assert tuple(model_state_sha256(module) for module in modules) == before
