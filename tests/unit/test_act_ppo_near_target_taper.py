"""Offline contracts for optional S1 exploration taper; no simulator is started."""

from dataclasses import replace
import hashlib
import json

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
    policy_action_bounds,
    sample_bounded_policy_actions,
)
from so101_capstone.training.act_ppo_exploration_profile import (
    final_good_act_exploration_offset,
    final_good_act_exploration_signature,
    final_good_act_near_target_exploration_offset,
    final_good_act_near_target_exploration_signature,
)
from so101_capstone.training.act_ppo_gazebo import (
    _act_ppo_config,
    _execution_signature,
    _exploration_stage,
)
from so101_capstone.training.residual_ppo import model_state_sha256
from test_act_ppo import TinyActPolicy
from test_act_ppo_distance_exploration import HALF_SPANS, _exploration
from test_act_ppo_stage_cli import build_config
from test_stage_process_integration import PROJECT, REWARD, small_ppo_config, training_config


TARGET_PHASES = (
    ("APPROACH", "moving"), ("GRASP_LIFT", "manipulation"),
    ("TRANSPORT", "moving"), ("PLACE_RELEASE", "manipulation"),
)
HOME_PHASES = (("WAIT_HOME", "idle"), ("RETURN_HOME", "moving"), ("HOME_HOLD", "home_hold"))


def offset(info, stage="moving"):
    return final_good_act_near_target_exploration_offset(info, stage_name=stage, initial_log_std=-3.)


def physical_scales(info, stage="moving"):
    return _exploration().with_offset(offset(info, stage)).detach().double().exp().numpy() * HALF_SPANS


@pytest.fixture
def launch_config(tmp_path):
    stages = StageTrainingConfig.load(PROJECT / "config/act_ppo_stages_s1_progress.json")
    return replace(
        training_config(tmp_path, stages, "near_target"),
        exploration_profile="final_good_act_near_target", initial_log_std=-3.,
        minimum_log_std=-9., maximum_log_std=-2.5, gripper_maximum_log_std=-1.5,
    )


@pytest.mark.parametrize("phase,stage", TARGET_PHASES)
@pytest.mark.parametrize("distance,expected", [
    (0., .012), (.01, .012), (.02, .012), (.03, .0185), (.04, .025),
    (.045, .03125), (.05, .0375), (.06, .05), (.07, .065), (.08, .08),
    (.115, .1), (.15, .12), (.5, .12),
])
def test_all_s1_target_phases_use_interpolated_physical_scale_on_all_six_axes(phase, stage, distance, expected):
    np.testing.assert_allclose(physical_scales({
        "training_stage": "S1", "task_phase": phase, "stage_reach_distance_m": distance,
    }, stage), [expected] * 6, rtol=6e-7)


def test_distance_taper_is_continuous_and_monotonic_including_grasp_phase_boundary():
    distances = np.linspace(0., .2, 401)
    scales = np.stack([physical_scales({
        "training_stage": "S1", "task_phase": "APPROACH" if d > .045 else "GRASP_LIFT",
        "stage_reach_distance_m": float(d),
    }, "moving" if d > .045 else "manipulation") for d in distances])
    assert np.all(np.diff(scales, axis=0) >= 0.)
    for boundary in (.02, .04, .045, .06, .08, .15):
        pair = [physical_scales({"training_stage": "S1", "task_phase": "APPROACH",
                                 "stage_reach_distance_m": boundary + delta})
                for delta in (-1e-7, 1e-7)]
        np.testing.assert_allclose(pair[0], pair[1], atol=4e-7, rtol=0.)


@pytest.mark.parametrize("phase,stage", HOME_PHASES)
@pytest.mark.parametrize("training_stage", ["S0", "S1", None])
@pytest.mark.parametrize("error", [0., .02, .04, .08, .1, .2])
def test_home_behavior_is_bitwise_identical_to_existing_profile(phase, stage, training_stage, error):
    info = {"task_phase": phase, "stage_home_error_rad": error,
            "home_joint_tolerance_rad": .02, "stage_reach_distance_m": .005}
    if training_stage is not None:
        info["training_stage"] = training_stage
    np.testing.assert_array_equal(offset(info, stage), final_good_act_exploration_offset(
        info, stage_name=stage, initial_log_std=-3.,
    ))


@pytest.mark.parametrize("phase,stage", TARGET_PHASES)
@pytest.mark.parametrize("training_stage", [None, "S0", "S2", "full"])
@pytest.mark.parametrize("distance", [.02, .05, .1, .2])
def test_non_s1_full_task_behavior_is_bitwise_identical(phase, stage, training_stage, distance):
    info = {"task_phase": phase, "target_distance_m": distance}
    if training_stage is not None:
        info["training_stage"] = training_stage
    if training_stage == "S2":
        # A prepared S2 segment owns manipulation exploration even while the
        # full-task tracker still reports the preceding physical phase.
        stage = "manipulation"
    np.testing.assert_array_equal(offset(info, stage), final_good_act_exploration_offset(
        info, stage_name=stage, initial_log_std=-3.,
    ))


def test_existing_profile_s1_behavior_and_signature_hash_stay_unchanged():
    info = {"training_stage": "S1", "task_phase": "GRASP_LIFT", "stage_reach_distance_m": .02}
    original = final_good_act_exploration_offset(info, stage_name="manipulation", initial_log_std=-3.)
    np.testing.assert_allclose(_exploration().with_offset(original).detach().double().exp().numpy() * HALF_SPANS,
                               [.12] * 6, rtol=6e-7)
    encoded = json.dumps(final_good_act_exploration_signature(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == "7830d97fd382f44e219d761bc4c0f37e183cba55b5ac066d6e13ef14426bf3d2"


def test_present_stage_distance_takes_precedence_and_invalid_evidence_is_rejected():
    info = {"training_stage": "S1", "task_phase": "APPROACH", "target_distance_m": .2}
    np.testing.assert_allclose(physical_scales(info), [.12] * 6, rtol=6e-7)
    np.testing.assert_allclose(physical_scales({**info, "stage_reach_distance_m": .02}), [.012] * 6, rtol=6e-7)
    for invalid in (None, float("nan"), float("inf"), -.01, True, "0.02"):
        with pytest.raises(ValueError, match="stage_reach_distance_m"):
            offset({**info, "stage_reach_distance_m": invalid})
    with pytest.raises(ValueError, match="does not match"):
        offset(info, "idle")


@pytest.mark.parametrize("maximum,gripper_maximum", [(-3.5, -1.5), (-2.5, -3.5)])
def test_collector_rejects_bounds_that_clip_taper_maximum(launch_config, maximum, gripper_maximum):
    invalid = replace(launch_config, initial_log_std=-4., maximum_log_std=maximum,
                      gripper_maximum_log_std=gripper_maximum)
    with pytest.raises(ValueError, match="bounds would clip"):
        _exploration_stage({"training_stage": "S1", "task_phase": "APPROACH",
                            "stage_reach_distance_m": .2}, invalid)


@pytest.mark.parametrize("near_limit", [False, True])
def test_actual_bounded_sampling_replays_saved_step_offsets_with_finite_likelihood(launch_config, near_limit):
    ppo = _act_ppo_config(launch_config, 1)
    exploration = ActPpoExploration(ppo)
    means = torch.tensor([99., -99., 99., -99., 99., 99.] if near_limit else
                         [0.] * 5 + [50.], dtype=torch.float64).expand(4096, 6)
    unchanged_means = means.clone()
    command_scales = []
    for distance, expected in ((.02, .012), (.08, .08), (.15, .12)):
        _, _, recorded_offset = _exploration_stage({
            "training_stage": "S1", "task_phase": "GRASP_LIFT", "stage_reach_distance_m": distance,
        }, launch_config)
        recorded_offset = np.asarray(recorded_offset, dtype=np.float32).copy()
        log_std = exploration.with_offset(recorded_offset)
        np.testing.assert_allclose(log_std.detach().double().exp().numpy() * HALF_SPANS,
                                   [expected] * 6, rtol=6e-7)
        torch.manual_seed(1042)
        actions, recorded_log_prob, latent = sample_bounded_policy_actions(means, log_std, return_latent=True)
        # PPO evaluates the recorded transition's offset, even if later distance changed.
        replay_log_std = exploration.with_offset(recorded_offset)
        replay_log_prob, entropy = evaluate_bounded_policy_actions(
            means, actions, replay_log_std, policy_action_latents=latent,
        )
        torch.testing.assert_close(replay_log_prob, recorded_log_prob, rtol=0., atol=0.)
        assert bool(torch.isfinite(replay_log_prob).all())
        assert bool(torch.isfinite(entropy).all())
        lower, upper = policy_action_bounds(device=actions.device, dtype=actions.dtype)
        assert bool(((actions >= lower) & (actions <= upper)).all())
        torch.testing.assert_close(means, unchanged_means, rtol=0., atol=0.)
        commands = np.stack([policy_positions_to_gazebo(row) for row in actions.detach().numpy()])
        command_scales.append(commands.std(axis=0))
        if not near_limit:
            np.testing.assert_allclose(command_scales[-1], [expected] * 6, rtol=.04)
    assert np.all(command_scales[0] < command_scales[1])
    assert np.all(command_scales[1] < command_scales[2])


def test_signature_records_only_new_profile_taper_and_returns_independent_values():
    signature = final_good_act_near_target_exploration_signature()
    assert signature["profile"] == "final_good_act_near_target"
    assert signature["sample_distribution"] == "bounded_mean_truncated_gaussian"
    assert signature["action_mean"] == "unchanged_absolute_act"
    assert "s1_near_override" not in signature
    taper = signature["s1_target_taper"]
    assert taper["training_stage"] == "S1"
    assert taper["phases"] == ["APPROACH", "GRASP_LIFT", "PLACE_RELEASE", "TRANSPORT"]
    assert [(item["distance_m"], item["initial_joint_std_rad"]) for item in taper["control_points"]] == [
        (.02, .012), (.04, .025), (.06, .05), (.08, .08), (.15, .12),
    ]
    taper["control_points"][0]["initial_joint_std_rad"] = 999.
    assert final_good_act_near_target_exploration_signature()["s1_target_taper"]["control_points"][0]["initial_joint_std_rad"] == .012


@pytest.mark.parametrize("old_is_source", [False, True])
def test_resume_rejects_changed_profile_before_modifying_any_module(launch_config, tmp_path, old_is_source):
    new_signature = _execution_signature(launch_config, REWARD)
    old_signature = _execution_signature(replace(launch_config, exploration_profile="final_good_act"), REWARD)
    source, target = (old_signature, new_signature) if old_is_source else (new_signature, old_signature)
    ppo = small_ppo_config(launch_config)

    def trainer(signature):
        return ActPpoTrainer(TinyActPolicy(chunk_size=1), ActPpoValueNetwork(ppo),
                             config=ppo, execution_signature=signature)

    original = trainer(source)
    path = original.save_checkpoint(tmp_path / "training_state.pt")
    matching = trainer(source)
    matching.load_checkpoint(path, map_location="cpu")
    assert model_state_sha256(matching.policy) == model_state_sha256(original.policy)
    incompatible = trainer(target)
    modules = (incompatible.policy, incompatible.critic, incompatible.exploration)
    before = tuple(model_state_sha256(module) for module in modules)
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="execution signature"):
        incompatible.load_checkpoint(path, map_location="cpu")
    assert tuple(model_state_sha256(module) for module in modules) == before
    assert torch.equal(torch.get_rng_state(), before_rng)


@pytest.mark.parametrize("multiworld", [False, True])
def test_both_cli_entrypoints_route_opt_in_profile_to_physical_exploration(multiworld):
    config = build_config([
        "--reward-profile", "pick_place_v2", "--exploration-profile", "final_good_act_near_target",
        "--training-stages", "s1", "--initial-log-std", "-3", "--maximum-log-std", "-2.5",
        "--gripper-maximum-log-std", "-1.5", "--final-target-absent-s", "0",
    ], multiworld=multiworld)
    # build_config's single-world helper intentionally maps only reward kwargs;
    # supply the unchanged shared numeric bounds for the selector comparison.
    config = replace(config, initial_log_std=-3., minimum_log_std=-9., maximum_log_std=-2.5,
                     gripper_maximum_log_std=-1.5)
    assert config.exploration_profile == "final_good_act_near_target"
    stage, _, selected = _exploration_stage({"training_stage": "S1", "task_phase": "GRASP_LIFT",
                                            "stage_reach_distance_m": .02}, config)
    assert stage == "manipulation"
    logs = ActPpoExploration(_act_ppo_config(config, 1)).with_offset(selected)
    np.testing.assert_allclose(logs.detach().double().exp().numpy() * HALF_SPANS, [.012] * 6, rtol=6e-7)
