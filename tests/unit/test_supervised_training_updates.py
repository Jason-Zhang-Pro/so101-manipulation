"""CPU checks of supervised accumulation and exact logged update numbers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.scripts.train import update_policy
from lerobot.utils.logging_utils import MetricsTracker


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0
        self.update_calls = 0

    def forward(self, batch):
        self.forward_calls += 1
        loss = ((self.weight * batch["x"] - batch["y"]) ** 2).mean()
        return loss, {"mse": loss.detach().item()}

    def update(self):
        self.update_calls += 1


def run_update(policy, batches, accumulation_steps):
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    metrics = SimpleNamespace()
    _, output = update_policy(
        metrics, policy, batches[0], optimizer, 100.0,
        torch.amp.GradScaler("cpu", enabled=False),
        lr_scheduler=scheduler,
        gradient_accumulation_steps=accumulation_steps,
        batch_iterator=iter(batches[1:]),
    )
    metrics.output = output
    return metrics, scheduler


@pytest.mark.parametrize("tail_size", [1, 2])
def test_accumulated_microbatches_match_one_combined_optimizer_update(tail_size):
    batches = [
        {"x": torch.tensor([1.0, 2.0]), "y": torch.tensor([2.0, -1.0])},
        {"x": torch.tensor([3.0, -2.0])[:tail_size], "y": torch.tensor([0.0, 4.0])[:tail_size]},
    ]
    accumulated, combined = TinyPolicy(), TinyPolicy()
    micro_metrics, scheduler = run_update(accumulated, batches, 2)
    batch_metrics, _ = run_update(
        combined, [{key: torch.cat([b[key] for b in batches]) for key in batches[0]}], 1
    )
    torch.testing.assert_close(accumulated.weight, combined.weight)
    assert micro_metrics.loss == pytest.approx(batch_metrics.loss)
    assert micro_metrics.output["mse"] == pytest.approx(batch_metrics.output["mse"])
    assert accumulated.forward_calls == 2
    assert accumulated.update_calls == 1
    assert scheduler.last_epoch == 1
    assert accumulated.weight.grad is None


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_accumulation_is_rejected_before_update(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        run_update(TinyPolicy(), [{"x": torch.ones(1), "y": torch.ones(1)}], invalid)


def test_tensorboard_parser_preserves_distinct_update_numbers():
    path = Path(__file__).resolve().parents[2] / "scripts/lerobot_log_to_tensorboard.py"
    spec = importlib.util.spec_from_file_location("so101_tb_log", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracker = MetricsTracker(2, 100, 5, {}, initial_step=1010)
    assert module.parse_step(str(tracker)) == 1010
    tracker.steps = 1490
    assert module.parse_step(str(tracker)) == 1490
