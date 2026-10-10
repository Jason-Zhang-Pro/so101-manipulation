# ACT: Collection to Real-Robot Evaluation

Run commands from the repository root in the existing compatible LeRobot Python
environment. All collect/train/deploy commands default to preview. Add `--execute`
only when ready to run that stage. Evaluation of a filled trial CSV is read-only.

## 1. Configure hardware and data

Edit `configs/act_workflow.json`. Set the follower/leader USB ports, camera device
paths, IDs, dataset repo ID and local root. Prefer stable `/dev/v4l/by-id/` paths
when available. Camera names `fixed` and `handeye` must match the checkpoint.
Dataset FPS is 15 for the documented black15 baseline; camera capture requests
30 FPS. Those are different settings, and actual device support must be checked.

All relative data/model paths resolve from this repository root. Raw data and
outputs are gitignored. Preserve the existing LeRobot calibration associated with
the configured robot IDs; this workflow does not recalibrate hardware.

## 2. Collect demonstrations

```bash
python learning/act/workflow.py collect
python learning/act/workflow.py collect --episodes 10 --execute
python learning/act/workflow.py collect --append --episodes 10 --execute
```

The first command previews the collection plan. Subsequent examples create and
then append demonstrations. Existing datasets are never deleted automatically.
For 15-point experiments, record the intended P0-P14 label per episode while
collecting; record grasp success, approach style, lighting and any corrections.
Keep a consistent starting pose and verify both images before recording.

## 3. Validate the dataset

```bash
python evaluation/check_dataset.py --root datasets/so101_black15
```

Checks per-episode timestamps, frame indices, finite action/state values, metadata
lengths and complete video decode. These are structural checks, not a judgement
of successful grasping or demonstration style. Inspect flagged episodes and
visually review successful grasp/placement before deciding to train.

## 4. Train ACT

```bash
python learning/act/workflow.py train --run act_black15_baseline
python learning/act/workflow.py train --run act_black15_baseline --execute
python learning/act/workflow.py tensorboard --run act_black15_baseline --execute
```

Training uses explicit local dataset root, ImageNet ResNet18, chunk 50, one action
step with temporal ensemble, KL 0.5, batch 8, seed 1000 and 100000 updates.
Checkpoints are saved every 2000 updates. The output directory is created by
LeRobot, not pre-created by the launcher. Image augmentation defaults to off to
match the inspected baseline; set `image_augmentation` in the workflow config for
a separate controlled run. Learning rates are set on the ACT preset itself so
LeRobot's preset validation does not silently replace a CLI optimizer override.

TensorBoard opens at `http://127.0.0.1:6006`. Log-derived metrics are always
available under `log_scalars` when the trainer emits the expected log format.
The locally modified trainer adds richer metrics; see `environment.md`.
Latent monitoring remains available through `act_latent_to_tensorboard.py`.

Resume the exact checkpoint, including optimizer state:

```bash
python learning/act/workflow.py resume --checkpoint outputs/train/act_black15_baseline/checkpoints/050000/pretrained_model --steps 100000 --execute
```

Resume requires the companion training-state files in the checkpoint parent.
Copying only weights is sufficient for inference, not for an exact training resume.

## 5. Deploy on the real follower

```bash
python learning/act/workflow.py deploy --checkpoint outputs/train/act_black15_baseline/checkpoints/100000/pretrained_model --run seen_eval
python learning/act/workflow.py deploy --checkpoint outputs/train/act_black15_baseline/checkpoints/100000/pretrained_model --run seen_eval --execute
```

Deployment validates camera keys and resolution, then uses LeRobot's real follower
backend. It displays images, records evaluation video/data, predicted actions and
observed qpos. Each new evaluation uses a fresh output path; no `rm -rf` is needed.
Run names are explicit and existing evaluation folders cannot be overwritten.
Reset time allows manual repositioning; it does not automatically command a home
pose. The legacy follower uses an unlimited `max_relative_target=null`; inspect
the first command and workspace before execution. The action log contains policy
targets and measured observations, not proof that each target was reached.

## 6. Record and summarize outcomes

```bash
python learning/act/workflow.py trial-sheet --trial-file outputs/evaluation/seen_eval/trials.csv --execute
python learning/act/workflow.py evaluate --trial-file outputs/evaluation/seen_eval/trials.csv
```

The sheet has 10 planned trials per P0-P14. Fill `grasp_success` with 0/1 after
reviewing each run; fill `place_success` only when placement was evaluated. Match
episode indices to actual recordings. For unseen tests, change the cell label and
`position_set` to `unseen`. Record real miss directions rather than plot offsets.
Summary groups by point, seen/unseen and lighting, and provides grasp rate and
Wilson confidence intervals. Full-task success is reported separately only for
rows with placement outcomes. No simulated or invented outcomes are included.
