# Third-Party Attribution and Distribution Review

- ACT is used through LeRobot, rather than reimplemented as an original algorithm.
- The historical reach PPO baseline uses Stable-Baselines3. The imported
  Capstone module implements direct ACT-PPO and retains its source attribution.
- ROS 2, Gazebo and MoveIt provide middleware, simulation and planning components.
- Robot meshes/URDFs include imported or generated SO-101 assets. Their original
  repository, authors and redistribution terms need confirmation.
- The MoveIt package declares BSD in `package.xml`; this alone does not establish
  a license for every mesh or all project scripts.

Upstream references for attribution:

- https://github.com/huggingface/lerobot
- https://github.com/DLR-RM/stable-baselines3
- https://github.com/moveit/moveit2

This publication draft carries no blanket license. Confirm team authorization,
source provenance and asset redistribution terms before public release.

Integration sources are pinned in `docs/integration_sources.json`. LeRobot's
Apache-2.0 license and copyright headers are retained in `lerobot/`. Imported
ROS package licenses remain alongside those packages; they do not establish a
blanket license for the combined repository. Capstone source was obtained from
a private repository and has only been integrated locally at this stage.
