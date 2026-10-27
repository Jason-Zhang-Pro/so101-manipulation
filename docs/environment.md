# Observed LeRobot Environment

> Integration update: the active shared source is now `lerobot/`. This page
> preserves the original environment snapshot; see `integration_zh.md` for the
> combined runtime and `integration_sources.json` for pinned source commits.

Local LeRobot base commit inspected: `882c80d446a63a44868c67ae535467af32ce0e80`.
This checkout has uncommitted changes to the camera implementation, camera utils,
policy config, motor bus and trainer. A clean checkout at this commit does not
exactly reproduce the working machine. The five source-file changes are now
captured in `patches/lerobot-local.patch`; see `lerobot_reproduction.md` for
application instructions, limitations and verification. Binary test artifacts
are deliberately excluded.

Observed Python packages (a partial environment snapshot, not a complete lock):

| Package | Version |
|---|---|
| torch | 2.7.1 |
| torchvision | 0.22.1 |
| draccus | 0.10.0 |
| huggingface-hub | 1.25.1 |
| numpy | 2.2.6 |
| opencv-python | 4.13.0.92 |
| tensorboard | 2.20.0 |
| safetensors | 0.8.0 |

The workflow invokes installed `lerobot-record` and `lerobot-train` commands.
Use the existing compatible environment for now. The offline CLI itself uses
only standard-library modules; video auditing requires PyAV, NumPy and PyArrow.

The local trainer honors `LEROBOT_TENSORBOARD_DIR`. The workflow also pipes its
log through the included converter to produce loss/grad_norm/lr on installs
without that trainer modification. Rich latent statistics use the separate
checkpoint sidecar. The pipeline does not promise live z values from every step.
