# Source Inventory and Organization

The allowlist contains 71 original files. Exact paths, byte counts and original
SHA-256 hashes are recorded in [source_manifest.json](source_manifest.json).

| Original location | Publication location | Treatment |
|---|---|---|
| `so101_description/` | `ros/src/so101_description/` | Robot package and planning mesh set |
| `so101_moveit_config/` | `ros/src/so101_moveit_config/` | Generated MoveIt configuration |
| `worlds/` | `simulation/worlds/` | Gazebo scene and relative visual assets |
| `so101_ros2/controllers.yaml` | `simulation/config/controllers.yaml` | Controller configuration |
| `Control so101/classic_grasp_cube.py` | `control/` | Visual grasp entry point |
| Three Python files in `so101/` | `learning/ppo/` | Existing reach baseline |
| Selected general ACT utilities in `tools/` | `learning/act/` | Deployment, logging and latent tools |
| Calibration/reset utilities in `tools/` | `tools/` | Runtime utilities |
| `tools/create_fixed_only_dataset.py` | `evaluation/` | Camera ablation dataset utility |

Gazebo and MoveIt keep separate model representations because they serve
different consumers. Mesh copies in those packages preserve the existing
resource-resolution conventions; no symlinks point into the working directory.

Publication-only corrections:

- Replaced machine-specific LeRobot source injection with optional `LEROBOT_SRC`.
- Made the ACT playback dataset root explicit.
- Replaced hard-coded world/controller paths with runtime-generated paths.
- Removed accidentally concatenated duplicate PPO code after the first complete
  training program. The original working file remains untouched.

New materials: README, evidence and contribution notes, review checklist,
reported-result fixture, three figure assets, figure renderer, offline checker
and world preparation script.

The subsequent ACT integration adds a unified standard-library CLI, workflow
JSON configuration, read-only dataset checker, trial-sheet evaluator, offline
regression checks and bilingual workflow/environment documentation. It invokes
the installed LeRobot implementation; no model weights or raw videos are copied.
