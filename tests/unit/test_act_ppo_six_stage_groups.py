"""Six-stage credit/filtering with unchanged GAE and legacy resume support."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from so101_capstone.training.act_ppo import (
    ActPpoTrainer, ActPpoValueNetwork, _execution_signature, configure_bounded_act_policy,
)
from test_act_ppo import TinyActPolicy
from test_act_ppo_stage_advantages import config, buffer_for


def test_six_groups_normalize_separately_without_changing_value_targets():
    cfg = config(advantage_group_count=6)
    groups = np.repeat(np.arange(6), 3)
    rewards = np.concatenate([[10**s, 2 * 10**s, 3 * 10**s] for s in range(6)])
    enabled = [True, True, False] * 6
    buffer = buffer_for(groups, rewards, cfg=cfg, enabled=enabled)
    batch = buffer.batch("cpu")
    original = buffer_for(groups, rewards, enabled=enabled,
                          cfg=replace(cfg, advantage_group_context_index=None)).batch("cpu")
    torch.testing.assert_close(batch.returns, original.returns, rtol=0, atol=0)
    for group in range(6):
        assert batch.advantages[group * 3:group * 3 + 3].tolist() == [-1., 1., 0.]
        assert batch.advantage_normalization["groups"][str(group)]["actor_transition_count"] == 2


def test_excluding_s2_and_s5_removes_whole_episodes_without_gae_contamination():
    cfg = config(advantage_group_count=6, gamma=.9, gae_lambda=.8)
    groups = [2, 2, 3, 3, 5, 5, 0, 0]
    source = buffer_for(groups, [1e6, -1e6, 2, 3, 1e6, -1e6, 4, 5], cfg=cfg,
                        values=[.2] * 8, next_values=[.5] * 8,
                        terminated=[False] * 8, episode_end=[False, True] * 4)
    old_adv, old_returns = source._advantages_and_returns()
    kept = source.without_stage_groups({2, 5})
    new_adv, new_returns = kept._advantages_and_returns()
    np.testing.assert_array_equal(new_adv, old_adv[[2, 3, 6, 7]])
    np.testing.assert_array_equal(new_returns, old_returns[[2, 3, 6, 7]])
    assert kept.episode_end == [False, True, False, True]
    assert [row[1] for row in kept.critic_task_context] == [3, 3, 0, 0]
    assert len(source) == 8
    assert not source.without_stage_groups(set(range(6)))


@pytest.mark.parametrize("group", [-1, 6, 2.5, float("nan")])
def test_six_stage_invalid_ids_cannot_enter_buffer(group):
    with pytest.raises(ValueError):
        buffer_for([group], [1], cfg=config(advantage_group_count=6))


def test_six_stage_signature_and_binary_legacy_checkpoint(tmp_path):
    policy = TinyActPolicy()
    configure_bounded_act_policy(policy)
    binary = _execution_signature(policy, config(), {})["advantage_normalization"]
    categorical = _execution_signature(policy, config(advantage_group_count=6), {})["advantage_normalization"]
    assert binary["kind"] == "binary_context_actor_only_v1"
    assert "group_count" not in binary
    assert categorical["kind"] == "categorical_context_actor_only_v1"
    assert categorical["group_count"] == 6
    cfg = config()
    trainer = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg)
    payload = trainer.checkpoint_payload()
    payload["config"].pop("advantage_group_count")
    path = tmp_path / "old_binary.pt"
    torch.save(payload, path)
    restored = ActPpoTrainer(TinyActPolicy(), ActPpoValueNetwork(cfg), config=cfg)
    restored.load_checkpoint(path)
    assert trainer.execution_signature == restored.execution_signature
