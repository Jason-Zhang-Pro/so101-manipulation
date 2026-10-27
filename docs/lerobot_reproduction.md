# Reproduce the Local LeRobot Changes

> Historical reproduction instructions. The integrated checkout already contains
> these selected changes in `lerobot/`, plus ACT-PPO and latent logging changes.
> Do not apply this patch to the integrated source. See `integration_zh.md`.

Base commit: `882c80d446a63a44868c67ae535467af32ce0e80`.
The patch records the current working-tree source changes, not new algorithm
changes. It excludes binary test fixtures, datasets, checkpoints and credentials.
Upstream LeRobot licensing continues to apply to these modifications.

## Apply in a Separate Checkout

Run from the publication repository root. Do not apply over a dirty checkout.

```bash
RELEASE_ROOT="$PWD"
git clone https://github.com/huggingface/lerobot.git ../lerobot-reproduction
cd ../lerobot-reproduction
git checkout 882c80d446a63a44868c67ae535467af32ce0e80
git apply --check "$RELEASE_ROOT/patches/lerobot-local.patch"
git apply "$RELEASE_ROOT/patches/lerobot-local.patch"
python -m pip install -e .
python -m pip install tensorboard==2.20.0
```

Use a separate Python 3.10 environment. See `environment.md` for observed
versions; that snapshot is not a tested complete dependency lock. Choose Torch
and its CUDA build for the target machine. ROS imports additionally require the
ROS environment described in `setup.md`.

## Changes and Consequences

| File | Local behavior | Caveat |
|---|---|---|
| cameras/opencv/camera_opencv.py | Validate read-back FPS and dimensions even when OpenCV set returns false | Does not remove FPS checks or measure achieved frame delivery |
| cameras/utils.py | Windows uses DirectShow | Linux unchanged; original patch retains an unreachable MSMF return |
| configs/policies.py | Closed temporary JSON before loading; accepts gradient_accumulation_steps; selects features by type | Field alone does not implement accumulation; canonical ACT keys are still recommended |
| motors/motors_bus.py | One retry by default for sync reads/writes; retry failures logged at INFO | Failed communication can increase latency; not a control-safety guarantee |
| scripts/train.py | SummaryWriter logs numerical tracker/output metrics at log intervals | Requires tensorboard; does not add live latent histograms |

Set `LEROBOT_TENSORBOARD_DIR` to enable native training metrics. Rich latent
monitoring is implemented separately by `learning/act/act_latent_to_tensorboard.py`.
Do not interpret that sidecar as logging every optimizer step.

## Verification Scope

The patch passed `git apply --check` against source exported from the specified
base commit. This establishes patch applicability only. A clean dependency
installation, training run and physical-robot validation remain to be performed.
Original working-machine source files were not modified by publication work.
