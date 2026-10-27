# SO-101 Collision Proxy Generation

This directory records how the Gazebo collision geometry used by the SO-101
Capstone project was generated and validated.

The visual meshes are intentionally left unchanged. Only `<collision>` geometry
is replaced by low-complexity convex proxy meshes.

## Motivation

During Direct ACT PPO Gazebo finetuning, sustained rigid contact such as the
robot pressing into the table or the gripper closing on the cube can make
contact solving expensive. In earlier multi-world runs this correlated with
camera reuse / target-feedback freshness failures, which caused partial
episodes to be discarded.

The collision proxy work reduces contact-solver geometry cost without clipping
the policy action space or disabling robot-table / gripper-cube physics.

## Toolchain

Generation environment used during the 2026-08-09 simplification work:

- Python 3.10 (`lerobot` Conda environment)
- `coacd==1.0.11`
- `trimesh>=4.4` (the machine installed `trimesh 5.0.0`)
- NumPy
- Matplotlib

The scripts in this directory are the final versions used to create and inspect
the installed proxy geometry.

## Final strategy

### Ordinary robot collision meshes

Most original collision STL files are replaced by one aggressively simplified
convex hull per source STL.

Typical final complexity is approximately 42-44 triangle faces per hull.

The generation profile prioritizes very low collision cost:

- CoACD threshold: `1.0`
- maximum convex hulls: `1`
- preprocess mode: `auto`
- preprocess resolution: `25`
- MCTS nodes: `10`
- MCTS iterations: `60`
- MCTS max depth: `2`
- merge: enabled
- decimate: enabled
- maximum convex-hull vertices: `24`
- approximation mode: `ch`
- seed: `0`

### Fixed gripper side: wrist_roll_follower

The automatic two-hull decomposition was replaced by a functional partition:

1. one filled body hull;
2. one fixed gripping-finger hull.

The large servo-mounting recess is intentionally filled because it is internal
geometry and does not need to participate in the manipulation task.

Validated generated geometry:

- body: `44` faces;
- fixed finger: `42` faces;
- original fixed-jaw to 30 mm cube approximate gap: `0.000 mm`;
- generated finger to cube approximate gap: `0.000 mm`;
- generated body to cube approximate gap: `15.701 mm`.

### Moving jaw

The moving jaw is also split functionally:

1. one filled body hull;
2. one moving gripping-finger hull.

Validated generated geometry:

- body: `44` faces;
- moving finger: `44` faces;
- original 30 mm cube first-contact joint angle: about `-0.320 deg`;
- generated first-contact joint angle: about `-0.430 deg`;
- contact-angle difference: about `0.110 deg`;
- first contacting generated part: `finger`;
- body gap at first contact: about `14.302 mm`;
- finger approximate gap at detected contact: about `0.268 mm`.

The contact search used a `0.35 mm` diagnostic tolerance.

## Camera collision

`gripper_camera_mount.stl` was generated as an experimental proxy by the
general generator, but the first installed stress-test Xacro intentionally
keeps the original camera-mount collision.

The gripper camera body and lens already use primitive collision geometry.

Do not claim the generated camera-mount proxy is active unless the Xacro is
explicitly changed and revalidated.

## Installed assets

The runtime proxy meshes are installed in the source package at:

```text
ros2_ws/src/so101_description/meshes/collision_generated_ultralow/
```

The source Xacro references the active proxies from:

```text
ros2_ws/src/so101_description/urdf/so101.urdf.xacro
```

The manifest stored beside the generated meshes records generated proxy files.

## Scripts

`generate_so101_ultralow_collision_proxies.py`

- discovers collision STL use from the current Xacro;
- generates aggressive low-complexity proxies;
- creates reports and four-view overlay previews;
- does not modify project source.

`generate_so101_functional_wrist_follower_proxy.py`

- replaces the fixed gripper side with a functional body + finger partition;
- fills the unnecessary servo-mount recess;
- evaluates candidates relative to the 30 mm cube.

`generate_so101_functional_moving_jaw_proxy.py`

- replaces the moving jaw with a functional body + finger partition;
- compares the generated first-contact angle against the original STL.

`preview_so101_gripper_assembled_collision.py`

- assembles fixed jaw, moving jaw and the 30 mm cube using current Xacro joint
  geometry;
- provides open, 45-degree and automatic-grasp overlays.

`apply_so101_ultralow_collisions_for_stress_test.py`

- copies generated proxy assets into `so101_description`;
- patches only collision mesh references;
- preserves visual mesh references, origins, scales, joints and inertials;
- backs up the previous Xacro / generated collision directory under `/tmp`;
- does not run `git add` or `git commit`.

## Validation status

As of 2026-08-09:

- proxy generation completed successfully;
- assembled gripper overlays were visually inspected and accepted;
- `so101_description` rebuilt successfully after installation;
- single-world Gazebo was run with the proxy collision model and no
  recoverable camera/target freshness failure was observed before the run was
  manually interrupted;
- full long-duration four-world finetune validation is still pending.

Therefore the collision model is considered suitable for continued runtime
testing, but the README should be updated if later four-world testing reveals a
regression.

## Important constraints

- Do not replace visual meshes with these proxies.
- Do not enable action clipping merely to avoid physical contact.
- Do not disable robot-table or gripper-cube collision to hide solver problems.
- Keep `max_joint_delta_rad` disabled unless explicitly required by a later
  experiment.
- For grasp-critical geometry, preserve independent body/finger collision
  components rather than collapsing the entire gripper side into one hull.
