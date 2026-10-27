"""Check real launcher arguments and local ACT weights without starting training.

Only the assignment/argument-building sections of the base launcher are
evaluated. ROS/Gazebo, CUDA checks, cleanup and the trainer command are excluded.
"""
from dataclasses import fields
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    destination = root / "codex_workspace/reports/s0_s5_configuration_check"
    destination.mkdir(parents=True, exist_ok=True)
    resolved = subprocess.run(["bash", str(root / "scripts/finetune_act_ppo_s0_s5.sh"), "--print-config"],
                              check=True, text=True, capture_output=True).stdout
    base = (root / "scripts/finetune_act_ppo_so101.sh").read_text()
    prefix_anchor = 'source "$HOME/miniconda3/etc/profile.d/conda.sh"'
    tail_anchor = "ACTOR_LEARNING_RATE_ARGS=()"
    trainer = "python -m so101_capstone.training.act_ppo_multiworld_gazebo"
    assert all(base.count(anchor) == 1 for anchor in (prefix_anchor, tail_anchor, trainer))
    prefix = base.split(prefix_anchor)[0]
    tail = base[base.index(tail_anchor):].replace(trainer, "capture_training_arguments", 1)
    argv_file = destination / "argv.bin"
    capture = "\ncapture_training_arguments() { printf '%s\\0' \"$@\" > " + shlex.quote(str(argv_file)) + "; }\n"
    body = resolved + "\n" + prefix + "\nOUTPUT_DIRECTORY=" + shlex.quote(str(destination / "not_started")) + capture + tail
    completed = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", body],
                               cwd=root, check=True, text=True, capture_output=True)
    (destination / "resolved_shell.txt").write_text(resolved)
    (destination / "argument_build.log").write_text(completed.stdout)
    argv = argv_file.read_bytes().decode().rstrip("\0").split("\0")
    resolved_assignments = dict(token.split("=", 1) for token in shlex.split(resolved)
                                if "=" in token and not token.startswith("#"))
    os.environ["SO101_ASSISTED_GRASP_ENABLED"] = resolved_assignments["ASSISTED_GRASP_ENABLED"]
    from so101_capstone.training import act_ppo_multiworld_gazebo as multi
    from so101_capstone.training.act_ppo_gazebo import _resolved_stage_config, _act_ppo_config, _resolved_reward
    from so101_capstone.policies.act_checkpoint_loader import load_act_checkpoint
    args = multi._parser().parse_args(argv)
    base_config = multi._build_base_config(args)
    aliases = {"preflight_cleanup_enabled": "preflight_cleanup", "stage_success_skip_enabled": "stage_success_skip"}
    worlds = multi.MultiWorldConfig(**{field.name: getattr(args, aliases.get(field.name, field.name))
                                      for field in fields(multi.MultiWorldConfig)})
    stages = _resolved_stage_config(base_config)
    loaded = load_act_checkpoint(base_config.act_checkpoint, device="cpu", strict=True)
    ppo = _act_ppo_config(base_config, min(loaded.config.n_action_steps, base_config.act_max_action_steps))
    contract = multi._multiworld_execution_signature(base_config, _resolved_reward(base_config), worlds)
    result = dict(configuration_valid=True, weights_loaded_on_cpu=True, simulation_started=False,
        optimizer_created=False, actor_checkpoint=str(base_config.act_checkpoint),
        training_stages=base_config.training_stages, world_count=worlds.world_count,
        gazebo_gui=worlds.gazebo_gui, chunk_size=loaded.config.chunk_size,
        n_action_steps=loaded.config.n_action_steps, temporal_ensemble_coeff=loaded.config.temporal_ensemble_coeff,
        critic_task_context_dim=ppo.critic_task_context_dim, advantage_group_count=ppo.advantage_group_count,
        assisted_grasp_enabled=os.environ["SO101_ASSISTED_GRASP_ENABLED"] == "true",
        target_kl=ppo.target_kl, actor_learning_rate=ppo.actor_learning_rate,
        stage_budgets_s={f"S{i}": stages.stage_budget_s(f"S{i}") for i in range(6)},
        checkpoint_every_updates=base_config.checkpoint_every_updates)
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "execution_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
