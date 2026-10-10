# Third-Party Attribution and Distribution Review

- ACT is used through LeRobot, rather than reimplemented as an original algorithm.
- PPO uses Stable-Baselines3.
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
