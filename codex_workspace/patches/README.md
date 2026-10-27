> 发布副本说明：`lerobot/` 已包含当前全部定制源码，不要再次应用这里的补丁。完整基线与差异见 `docs/publication/source_provenance.json` 和 `lerobot_local_changes.patch`。

# Local LeRobot ACT bounded-output patch

The parent Capstone repository ignores `lerobot/`, which is a separate local Git
checkout. `act_bounded_output.patch` preserves the exact native ACT changes needed
by the bounded ACT-PPO implementation without tracking the entire vendor tree.
No dependencies or weights change.

The patch includes only:

- `src/lerobot/policies/act/configuration_act.py`
- `src/lerobot/policies/act/modeling_act.py`
- the new `src/lerobot/policies/act/bounded_actions.py`

Baseline: local LeRobot commit `feaea6a02218846e36fe034c2fabab82aa52eabb`.
Patch SHA-256: `20b7b49d83699e81a4b8719c9a0ff097c48697758f5145a77b46492479b4f0fb`.
The two existing ACT files had no earlier modifications in the inspected nested
Git diff. New Python files use mode `100644` in the patch; this avoids inheriting
the external volume's synthetic executable permission bits.

The patch is already applied in the current workspace. To verify this without
changing any file, run:

```bash
PROJECT_ROOT="/media/tiger/New Volume/ubuntufile/capstone"
git -c safe.directory="$PROJECT_ROOT/lerobot" -C "$PROJECT_ROOT/lerobot" \
    apply --reverse --check "$PROJECT_ROOT/codex_workspace/patches/act_bounded_output.patch"
```

For a separate fresh checkout at the matching baseline, check first and then
apply the patch from its local copy. Change `PROJECT_ROOT` to that checkout:

```bash
PROJECT_ROOT="/path/to/restored/capstone"
git -c safe.directory="$PROJECT_ROOT/lerobot" -C "$PROJECT_ROOT/lerobot" \
    apply --check "$PROJECT_ROOT/codex_workspace/patches/act_bounded_output.patch"
git -c safe.directory="$PROJECT_ROOT/lerobot" -C "$PROJECT_ROOT/lerobot" \
    apply "$PROJECT_ROOT/codex_workspace/patches/act_bounded_output.patch"
```

Run the second command only when the first succeeds. If it does not match, inspect
the actual local files and their changes; do not force, reset, or overwrite them.
The per-command `safe.directory` option does not modify global Git configuration.

Validation on 2026-10-02: reverse check against the modified workspace and forward
check against isolated copies of the baseline both succeeded. Native bounded ACT
tests passed: `22 passed`. These are offline checks; no Gazebo or training was run.

The config fields `action_bounds_lower` and `action_bounds_upper` default to
`None`, preserving historical checkpoint behavior. New bounded checkpoints store
both limits and `action_bound_softness=0.02`. Their helper soft-bounds raw physical
action means once, before queueing/temporal averaging. `clamp_action_mean_roundoff`
only guards arithmetic overshoot after averaging; it must not replace the soft
bound or apply that nonlinear transform a second time.
