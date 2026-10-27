"""Episode-level qualification, exactly-one-batch skips and safe checkpointing."""
import json
import math

import pytest

from so101_capstone.training.act_ppo_stage_success_skip import StageSuccessSkipController


def episode(stage="S1", success=True, **overrides):
    return dict(training_stage=stage, valid=True, stage_success=success, **overrides)


def observe_many(controller, stage, successes, failures=0):
    for success in [True] * successes + [False] * failures:
        controller.observe(episode(stage, success))


def test_eighty_percent_uses_all_valid_completed_episodes_not_transition_weight():
    controller = StageSuccessSkipController("s0_s1")
    for i in range(5):
        controller.observe(episode("S0", i < 4, steps=1 if i < 4 else 1000))
    observe_many(controller, "S1", successes=3, failures=2)
    decision = controller.finish_batch()
    assert decision.episode_counts == {"S0": 5, "S1": 5}
    assert decision.success_counts == {"S0": 4, "S1": 3}
    assert decision.success_rates == {"S0": .8, "S1": .6}
    assert decision.exclude_update_stages == {"S0"}
    assert decision.next_skip_stages == {"S0"}
    assert controller.active_stages == {"S1"}
    assert controller.episode_counts == {"S0": 0, "S1": 0}


def test_skipped_stage_cannot_requalify_without_data_and_returns_after_one_batch():
    controller = StageSuccessSkipController("s0_s1")
    observe_many(controller, "S0", 4, 1)
    observe_many(controller, "S1", 1, 4)
    controller.finish_batch()
    with pytest.raises(ValueError, match="skipped"):
        controller.observe(episode("S0"))
    observe_many(controller, "S1", 1, 4)
    decision = controller.finish_batch()
    assert decision.success_rates["S0"] is None
    assert decision.episode_counts["S0"] == 0
    assert decision.exclude_update_stages == set()
    assert controller.current_skipped_stages == set()
    assert controller.active_stages == {"S0", "S1"}
    assert controller.collection_batch_index == 2
    controller.observe(episode("S0"))


def test_other_stage_can_qualify_while_previously_skipped_stage_returns():
    controller = StageSuccessSkipController("s0_s1")
    controller.observe(episode("S0"))
    controller.observe(episode("S1", False))
    controller.finish_batch()
    observe_many(controller, "S1", 4, 1)
    decision = controller.finish_batch()
    assert decision.exclude_update_stages == {"S1"}
    assert controller.current_skipped_stages == {"S1"}
    assert controller.active_stages == {"S0"}


@pytest.mark.parametrize("selection,stages", [
    ("s0", ("S0",)), ("s1", ("S1",)), ("s0_s1", ("S0", "S1")),
])
def test_all_stage_skip_consumes_exactly_one_empty_batch(selection, stages):
    controller = StageSuccessSkipController(selection)
    for stage in stages:
        controller.observe(episode(stage))
    real = controller.finish_batch()
    assert real.batch_index == 0 and not real.empty_batch
    assert controller.active_stages == set()
    with pytest.raises(RuntimeError, match="consume_empty_batch"):
        controller.finish_batch()
    empty = controller.consume_empty_batch()
    assert empty.empty_batch and empty.batch_index == 1
    assert all(value == 0 for value in empty.episode_counts.values())
    assert all(value is None for value in empty.success_rates.values())
    assert not empty.exclude_update_stages and not empty.next_skip_stages
    assert controller.collection_batch_index == 2
    assert controller.active_stages == set(stages)
    with pytest.raises(RuntimeError, match="every configured"):
        controller.consume_empty_batch()


def test_invalid_episodes_are_neither_successes_nor_denominator_entries():
    controller = StageSuccessSkipController("s1")
    observe_many(controller, "S1", 4, 1)
    assert controller.observe({"training_stage": "S1", "valid": False}) is False
    assert controller.observe({"training_stage": "S1", "valid": False,
                               "stage_success": True}) is False
    decision = controller.finish_batch()
    assert decision.episode_counts["S1"] == 5
    assert decision.success_rates["S1"] == .8


def test_no_success_is_inferred_from_reward_or_generic_task_success():
    controller = StageSuccessSkipController("s1")
    record = episode(success=False, reward=100000,
                     task_phase="manipulation", stage_result="SUCCESS")
    record["success"] = True
    controller.observe(record)
    decision = controller.finish_batch()
    assert decision.success_rates["S1"] == 0
    assert not decision.exclude_update_stages


def test_unobserved_stage_has_no_rate_and_cannot_skip():
    controller = StageSuccessSkipController("s0_s1", threshold=.5)
    controller.observe(episode("S0"))
    decision = controller.finish_batch()
    assert decision.success_rates == {"S0": 1., "S1": None}
    assert controller.current_skipped_stages == {"S0"}


def test_invalid_only_batch_does_not_manufacture_a_success_rate():
    controller = StageSuccessSkipController("s1")
    controller.observe({"training_stage": "S1", "valid": False})
    decision = controller.finish_batch()
    assert decision.success_rates == {"S1": None}
    assert decision.exclude_update_stages == set()


@pytest.mark.parametrize("threshold,qualifies", [(.8, True), (.81, False), (1., False)])
def test_configurable_threshold_is_inclusive(threshold, qualifies):
    controller = StageSuccessSkipController("s1", threshold)
    observe_many(controller, "S1", 4, 1)
    assert bool(controller.finish_batch().exclude_update_stages) is qualifies


@pytest.mark.parametrize("bad", [0, -.1, 1.1, True, False, "0.8", None, math.inf, math.nan])
def test_rejects_invalid_thresholds(bad):
    with pytest.raises(ValueError, match="threshold"):
        StageSuccessSkipController("s1", bad)


@pytest.mark.parametrize("bad", ["full", "S1", "s6", "", None, ["s1"]])
def test_rejects_unsupported_selections(bad):
    with pytest.raises(ValueError, match="selection"):
        StageSuccessSkipController(bad)


@pytest.mark.parametrize("record", [
    None, [], {}, {"training_stage": "S2", "valid": True, "stage_success": True},
    {"training_stage": "S0", "valid": True, "stage_success": True},
    {"training_stage": ["S1"], "valid": True, "stage_success": True},
    {"training_stage": "S1", "valid": 1, "stage_success": True},
    {"training_stage": "S1", "valid": "true", "stage_success": True},
    {"training_stage": "S1", "valid": True},
    {"training_stage": "S1", "valid": True, "stage_success": "false"},
    {"training_stage": "S1", "valid": True, "stage_success": 1},
])
def test_malformed_record_is_rejected_without_opening_batch(record):
    controller = StageSuccessSkipController("s1")
    before = controller.state_dict()
    with pytest.raises(ValueError):
        controller.observe(record)
    assert controller.state_dict() == before


def test_invalid_episode_from_skipped_stage_is_still_contract_error():
    controller = StageSuccessSkipController("s0_s1")
    controller.observe(episode("S0"))
    controller.finish_batch()
    with pytest.raises(ValueError, match="skipped"):
        controller.observe({"training_stage": "S0", "valid": False})


def test_finish_requires_an_observed_batch():
    controller = StageSuccessSkipController("s1")
    with pytest.raises(RuntimeError, match="unobserved"):
        controller.finish_batch()
    controller.observe(episode(success=False))
    controller.finish_batch()
    with pytest.raises(RuntimeError, match="unobserved"):
        controller.finish_batch()


@pytest.mark.parametrize("valid", [True, False])
def test_partial_collection_cannot_be_checkpointed_or_overwritten(valid):
    controller = StageSuccessSkipController("s1")
    boundary = controller.state_dict()
    controller.observe({"training_stage": "S1", "valid": valid, "stage_success": False})
    with pytest.raises(RuntimeError, match="batch boundary"):
        controller.state_dict()
    with pytest.raises(RuntimeError, match="batch boundary"):
        controller.load_state_dict(boundary)
    controller.finish_batch()
    assert controller.state_dict()["collection_batch_index"] == 1


def test_checkpoint_round_trip_preserves_one_pending_skip_and_no_counts():
    original = StageSuccessSkipController("s0_s1")
    observe_many(original, "S1", 4, 1)
    decision = original.finish_batch()
    state = json.loads(json.dumps(original.state_dict()))
    restored = StageSuccessSkipController("s0_s1")
    restored.load_state_dict(state)
    assert restored.current_skipped_stages == {"S1"}
    assert restored.active_stages == {"S0"}
    assert restored.episode_counts == {"S0": 0, "S1": 0}
    assert restored.collection_batch_index == 1
    restored.observe(episode("S0", False))
    restored.finish_batch()
    assert restored.active_stages == {"S0", "S1"}
    assert decision.to_dict() == json.loads(json.dumps(decision.to_dict()))


def test_checkpoint_round_trip_preserves_all_stage_empty_skip():
    original = StageSuccessSkipController("s1")
    original.observe(episode())
    original.finish_batch()
    restored = StageSuccessSkipController("s1")
    restored.load_state_dict(original.state_dict())
    assert restored.consume_empty_batch().empty_batch
    assert restored.collection_batch_index == 2
    assert restored.active_stages == {"S1"}


@pytest.mark.parametrize("key,value", [
    ("schema", 2), ("schema", True), ("schema", 1.),
    ("selection", "s0"), ("selection", None),
    ("threshold", .9), ("threshold", True), ("threshold", math.nan),
    ("collection_batch_index", -1), ("collection_batch_index", True),
    ("collection_batch_index", 1.), ("collection_batch_index", "1"),
    ("current_skipped_stages", ["S0"]), ("current_skipped_stages", ["S1", "S1"]),
    ("current_skipped_stages", "S1"), ("current_skipped_stages", {"S1"}),
    ("current_skipped_stages", [None]), ("current_skipped_stages", [["S1"]]),
])
def test_malformed_checkpoint_is_rejected_atomically(key, value):
    controller = StageSuccessSkipController("s1")
    before = controller.state_dict()
    state = {**before, "collection_batch_index": 1, key: value}
    with pytest.raises(ValueError):
        controller.load_state_dict(state)
    assert controller.state_dict() == before


def test_rejects_partial_counts_unknown_keys_missing_fields_and_initial_skips():
    controller = StageSuccessSkipController("s1")
    valid = controller.state_dict()
    cases = [None, [], {**valid, "episode_counts": {"S1": 4}},
             {key: value for key, value in valid.items() if key != "schema"},
             {**valid, "current_skipped_stages": ["S1"]}]
    for bad in cases:
        with pytest.raises(ValueError):
            controller.load_state_dict(bad)
        assert controller.state_dict() == valid


def test_logs_states_and_counts_are_detached_copies():
    controller = StageSuccessSkipController("s1")
    state = controller.state_dict()
    state["current_skipped_stages"].append("S1")
    assert controller.current_skipped_stages == set()
    controller.observe(episode())
    counts = controller.episode_counts
    counts["S1"] = 999
    assert controller.episode_counts["S1"] == 1
    decision = controller.finish_batch()
    logged = decision.to_dict()
    logged["success_counts"]["S1"] = 1000
    assert decision.success_counts["S1"] == 1
    with pytest.raises(TypeError):
        decision.episode_counts["S1"] = 0
