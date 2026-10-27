# Direct ACT PPO numerical stability

This document describes sampling schema 4 and training-state schema 4. Both the
previous inverse-tanh distribution and schema 3's affine Gaussian with execution
clipping are historical implementations. Their training results do not validate
the bounded distribution introduced here. Offline tests do not establish Gazebo
or real-robot behavior.

The subsequent reward, curriculum, per-axis exploration and native ACT
replanning changes are documented in [reach learning](act_ppo_reach_learning.md).

## Action distribution and replay

ACT action targets use `[-100, 100]` for the five arm joints and `[0, 100]` for
the gripper. New PPO runs enable a differentiable bounded output transform after
ACT unnormalization, before chunk selection or temporal ensembling. Its bounds
and softness are saved in the ACT configuration, so standalone deployment and
differentiable PPO forwards use the same transform.

With the default normalized softness `s = 0.02`, the transform is exactly the
identity inside `[-0.98, 0.98]`. Outside this interval it uses smooth rational
tails approaching the endpoints. Values and first derivatives agree at the
join; finite out-of-range raw predictions retain a gradient instead of passing
through a hard clamp. This preserves the central part of existing ACT behavior,
including the previously observed HOME elbow target near 95.1, while removing
unbounded physical policy means. Predictions near a limit do change when the
transform is first enabled; a weights-only warm start is not behavior-identical
to the historical unbounded policy.

Exploration is a Gaussian truncated to the fixed legal interval in affine
action coordinates:

```text
centre = (upper + lower) / 2
scale = (upper - lower) / 2
normalized_mean = bounded_transform((raw_act_mean - centre) / scale)
sample ~ Normal(normalized_mean, exp(log_std)) conditioned on -1 <= sample <= 1
executed_action = centre + scale * sample
```

The rollout saves the actual bounded sample, old bounded ACT mean and effective
old log standard deviation. PPO evaluates that same sample with the truncated
Gaussian density, including its parameter-dependent normalization constant.
Support is fixed, so a legal recorded sample remains in support after an actor
update. Out-of-support samples have zero probability. Probability arithmetic
uses float64. Log probabilities and entropy use the affine coordinates; this
fixed coordinate choice leaves PPO ratios and KL unchanged.

The existing `mean_bounds_loss` configuration is retained for compatibility:

```text
mean_bounds_loss = mean(relu(abs(normalized_mean) - 1) ** 2)
mean_bounds_coefficient = 0.1
```

Because current physical means are bounded structurally, their bounds loss is
zero. The soft penalty is no longer responsible for enforcing legal means. This
change does not modify task rewards, learning rates, exploration schedules or
the configured target KL. Absolute joint safety checks remain in execution.

Deterministic execution uses the bounded ACT location parameter. Near a limit,
the mean of the truncated random distribution differs from this deterministic
location because outward noise is removed. Temporal ensembling, when configured,
combines already-bounded chunk predictions before sampling. Convex temporal
weights preserve the bounds, and replay recomputes all contributing predictions.
Ordinary ACT export retains its configured action schedule and temporal
ensembling.

Critic cached-plan schema 2 retains the normalized policy distribution locations;
these are now bounded. Its tensor layout is unchanged. The enclosing sampling
and training-state schemas distinguish the changed semantics. Episode step
records label their schema versions.

## Raw ACT overflow regularization

Legal executed actions do not imply that the network's pre-bound predictions
are well conditioned. For example, an arm prediction of 112 becomes 99.75, but
the rational tail's derivative is only 0.015625. The legacy bounded-mean loss
above cannot detect this drift.

`raw_mean_bounds_coefficient` adds a separate actor-only auxiliary loss using
the prediction **after unnormalization and before the soft bound**:

```text
raw_normalized = (raw_prediction - centre) / scale
raw_mean_bounds_loss = mean(relu(abs(raw_normalized) - 1) ** 2)
actor_loss += raw_mean_bounds_coefficient * raw_mean_bounds_loss
```

Legal raw predictions, including the HOME shoulder near -98.07, have zero loss.
This does not add a task reward penalty or pull valid joints toward zero/HOME.
It supplies a direct inward gradient for raw overflow without passing through
the flattened output transform. It is a soft regularizer, not a proof that all
future raw predictions are in range. The existing bounded transform, truncated
sampler and absolute joint checks still enforce execution limits.

Only predictions contributing to actual actor-enabled transitions are used.
Without temporal ensembling this selects the recorded chunk action index, not
every unused future prediction. With TE, each contributor is penalized before
weighting by its existing TE weight, so opposite overflows cannot cancel. ACT
singleton forwards and the bounded mean's numerical recurrence are unchanged;
there is no second ACT forward solely to obtain raw predictions.

The Python API and direct module CLI default to `0` for compatibility. The shell
launcher and `codex_workspace/commands/finetune_good40_posture_distance.sh`
enable `RAW_MEAN_BOUNDS_COEFFICIENT=0.1` as an initial trial setting. This is not a
validated optimal weight. Set the variable to `0` in the launcher, or edit that
explicit export in the saved command, to disable it. The old
`MEAN_BOUNDS_COEFFICIENT` remains a separate compatibility parameter.

The coefficient is saved in run/checkpoint configuration. Schema-4 checkpoints
missing it are interpreted as `0`; changing it on resume is allowed and prints
an explicit objective-change message. Other resume contracts remain strict.
The action distribution and inference/checkpoint schema do not change.

Each update records `raw_mean_bounds_loss`,
`raw_mean_out_of_bounds_fraction`, `raw_mean_overflow_normalized`, and
`raw_mean_bound_derivative_mean` in `metrics.jsonl`. TensorBoard uses
`loss/raw_mean_bounds_loss` and `action/raw_mean_*` for the other three fields.
These are actor-transition-weighted means from **accepted minibatch forwards**,
not final-policy evaluations or fresh episode counts. TE contributor statistics
are averaged with the TE weights. Rejected epochs do not contribute.
`raw_mean_bounds_diagnostic_count` counts actor transition presentations,
including repeated epochs. When this count is zero (disabled, no actor samples,
or no accepted epoch), raw traces are omitted instead of implying zero overflow.

## Probability consistency and ACT recomputation

Before the first optimizer step, the current actor must reproduce the collected
rollout probabilities and old bounded distribution parameters. Approximate KL,
clip fraction, maximum absolute log ratio and analytic truncated-Gaussian KL
should be zero up to the
configured numerical tolerance. A mismatch rejects the update before learning.

ACT forwards remain singleton computations per unique actor observation, with
the same evaluation-mode semantics used during collection. PPO transitions and
critic forwards can remain minibatched. Changing ACT batch size for throughput
requires repeating probability-consistency validation. Camera preprocessing,
temporal-history replay and episode rebuilding are unchanged by this numerical
correction.

## Learning rates and gradient clipping

The optimizer has separate ACT, backbone, critic and exploration parameter
groups. Exploration no longer inherits the actor's extremely small learning
rate. Gradient norms are clipped separately for actor, critic and exploration,
so a large critic gradient cannot scale down every actor update.

The following were historical schema-3 starting candidates, not an instruction
to override current launcher/environment values. This distribution change does
not alter them; inspect the current run's resolved configuration for actual
learning rates and KL threshold.

| Setting | Candidate | Status |
| --- | ---: | --- |
| Shell ACT learning rate | `1e-7` | Requires actual-rollout calibration |
| Shell backbone learning rate | `1e-8` | Requires actual-rollout calibration |
| Exploration learning rate | `1e-4` | Independent of ACT learning rate |
| Critic learning rate | `3e-4` | Retained starting value |
| Mean-bounds coefficient | `0.1` | Requires behavior validation |
| Entropy coefficient | `0` | No default entropy maximization |
| Analytic target KL | `0.02` | Requires actual-rollout calibration |

These are not validated optimal hyperparameters. The learning-rate inheritance
option is also distinct from a tuned recommendation. Parameter movement is
measured directly because nonzero gradients or a changed model hash alone do
not show how much of each parameter group actually changed in float32.

Exploration `log_std` is measured in the affine coordinates. Multiplying its
standard deviation by each axis's `scale` converts it to ACT action units;
the calibration then converts those units to radians. Per-axis executed noise
and proximity to limits remain necessary diagnostics: the untruncated scale
parameter is not the actual standard deviation of a truncated distribution near
a bound. Bounded output also does not guarantee successful or contact-free motion.

## Analytic KL, rollback and stopping

The epoch guard uses analytic truncated-Gaussian `KL(old || new)` on the fixed
legal support, summed over the six action dimensions and averaged over actor
transitions. Its reference is the policy that collected the current rollout,
not the original ACT checkpoint. The mean, 95th percentile and maximum are
reported separately. The configured target applies to the mean; it is not a
per-state or per-joint bound.

If an epoch crosses the target, the entire candidate epoch is rolled back:
actor, critic, exploration, optimizer state and CPU/CUDA/Python/NumPy RNG state.
With `kl_backoff_max_retries > 0`, the same epoch is retried with smaller actor,
backbone and exploration learning rates. The critic learning rate is unchanged.
The rollout probabilities, advantages and old-policy KL reference stay fixed;
the same minibatch permutation is replayed. A retry is accepted only when its
final full-rollout analytic KL satisfies the original target.

The Python API defaults to zero retries for compatibility. The launcher enables
at most five retries, multiplying policy learning rates by `0.5` each time, with
a minimum multiplier of `1e-4` relative to the configured rates. These values are
bounded trial settings, not proof of adequate ACT learning. A numerical failure
is rolled back immediately without repeating invalid computation. If retries
are exhausted or the multiplier cannot decrease, remaining epochs stop and
earlier accepted epochs remain.

The smallest actually attempted policy learning-rate multiplier persists into
subsequent updates and is saved in `learning_rate_state`. There is no automatic
increase. Reported actor/backbone/exploration rates are the effective rates.
Check parameter-changed fractions as well as KL: an accepted near-zero update
does not prove useful learning, particularly near float32 rounding limits.
Changing the minimum multiplier on resume does not silently raise an already
smaller saved rate.

Each restore clones the saved optimizer tensors before loading them. Reusing
the saved tensor objects can let later Adam steps mutate the rollback snapshot,
including CPU step counters for CUDA parameters. Regression tests cover retries
after the optimizer already has momentum, not only its first empty-state update.

Sample-based approximate KL and clip fraction remain diagnostic and do not
determine this guard. The guard is still an average bound, not a bound on every
state; phase-specific KL and calibrated command displacement reveal such tails.

## TensorBoard diagnostics

Each update writes the following additional traces:

- `loss/mean_bounds_loss`: the unweighted auxiliary mean-bounds loss.
- `ppo/truncated_gaussian_entropy`: entropy of the bounded distribution in
  affine action coordinates. Metrics retain `entropy` as a compatibility alias;
  TensorBoard emits only the distribution-specific trace.
- `ppo/pre_update_policy_kl`, `ppo/policy_kl`, `ppo/policy_kl_p95`,
  `ppo/policy_kl_max`: bounded-distribution consistency and accepted-policy
  movement. Structured metrics retain the old `pre_clip_policy_kl` field names
  as aliases, but current TensorBoard traces suppress those duplicates. Reading
  a historical payload with only old names preserves its old trace labels.
- `ppo/target_kl`, `ppo/configured_epochs`, `ppo/completed_epochs` and
  `ppo/early_stopped`: requested versus accepted optimization work.
- `optimization/{actor,critic,exploration}_gradient_norm`: each group's
  gradient norm before its own clipping.
- `optimization/{actor,critic,exploration}_parameter_relative_delta` and
  `optimization/{actor,critic,exploration}_parameter_changed_fraction`:
  accepted parameter movement after rollback, excluding parameters unused by
  the update. Relative delta is `||after - before|| / max(||before||, 1e-12)`;
  changed fraction counts elements whose stored values actually differ.
- `learning_rate/exploration`: the independent exploration learning rate.
- `exploration/log_std/joint_0` through `joint_5`, and matching
  `exploration/log_std_delta/joint_N`: each exploration parameter and its
  accepted change. Axis indices follow the existing six-joint action order.
- `ppo/kl_backoff_attempt_count`, `kl_backoff_rejection_count`,
  `kl_backoff_retry_count`, `kl_backoff_at_min_scale`, and
  `learning_rate/policy_scale`: attempted work versus accepted learning.
- `ppo/phase/<phase>/...`: accepted KL mean/p95/max and deterministic commanded
  position change, after execution clipping and conversion to calibrated radians.
  These are command changes, not measured physical robot motion. Context-free
  legacy data is labeled `unknown`. Every rejected candidate's corresponding
  diagnostics remain available in `metrics.jsonl` under `kl_backoff_attempts`.

Nonfinite diagnostic values are not written as TensorBoard scalars. A missing
point is not evidence that the value was zero; inspect the update rejection
message and structured metrics as well.

## Checkpoint migration

Sampling and complete PPO training states use schema 4. Schema 1, 2 or 3 PPO
training-state resume is rejected because the distribution and bounded-output
contract have changed. Old rollout actions, log probabilities and Gaussian
statistics must not be relabeled or reused as schema 4 data.

Existing compatible ACT weights remain usable as a warm start. Load their ACT
checkpoint, initialize fresh critic/exploration/optimizer and rollout state,
then collect new samples under the bounded transform. This is a new PPO run,
not a complete training-state resume. Loading an old checkpoint for ordinary
evaluation leaves its original behavior intact when bounds metadata is absent;
new PPO exports include that metadata and reproduce bounded inference after
reload. Existing checkpoint files are not rewritten during a warm start.

Within schema 4, `target_kl` and the three backoff control fields may be explicitly
overridden on resume. Other execution contracts, including reward and bounded
output settings, must match. Distribution schema versions and reward schema
versions identify different contracts and must not be confused.

The execution signature now includes the ACT forward contract: attention heads,
normalization, backbone, feature definitions and their order, action chunking,
temporal ensembling and other stored computation settings. Shape-compatible
configuration changes can therefore reject a resume even when tensor loading
would succeed. Device/path relocation and documented optimizer or publishing
metadata are not part of this forward contract; optimizer settings are checked
by the training-state machinery separately.

## Limits of this correction

These changes address probability modeling, numerical updates and observability.
They do not establish that a reach-only initial ACT can discover grasping,
lifting, transport, release and returning HOME. Reward coverage, exploration,
valid stage initialization and whole-task success still require separate design
and evaluation. This numerical correction does not modify task rewards or add a
curriculum, and its tests do not establish Gazebo or real-robot task success.

PPO clipping alone does not guarantee a small policy change; the role of an
additional KL check is described in [OpenAI Spinning Up's PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
The bounded retry mechanism here is an engineering extension to this project's
existing full-epoch rollback, not a claim that the reference implementation uses
the same procedure.
