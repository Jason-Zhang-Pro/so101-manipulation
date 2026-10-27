# Direct ACT PPO singleton actor-forward invariant

Established: 2026-08-09

## Permanent project rule

Direct ACT PPO rollout chunk generation evaluates ACT one observation at a
time (`batch=1`). PPO actor probability recomputation must use the same ACT
forward semantics.

`MINIBATCH_SIZE` remains the PPO transition minibatch size and may be greater
than 1. The critic, PPO loss aggregation, gradient calculation, and
`optimizer.step()` continue to use the normal transition minibatch. Only ACT
actor recomputation is evaluated one observation at a time and stacked back
into the transition minibatch.

Do not replace this singleton ACT recomputation with one batched ACT forward
as a performance optimization unless probability equivalence is revalidated
on the real CUDA ACT.

## Evidence from the 2026-08-09 diagnosis

Before any optimizer step, a real four-world rollout with 1350 actor
transitions produced:

- pre-update KL: `0.20919531`
- pre-update clip fraction: `0.015556` (about 21 / 1350 transitions)
- maximum absolute log-ratio: `263.16757`

A real-checkpoint CUDA comparison of ACT `batch=1` versus `batch=30` on the
same observations measured:

- maximum selected mean drift: about `0.005267`
- effective `log_std=-6`: ordinary max `|log_ratio| ~= 0.0763`
- effective `log_std=-4`: ordinary max `|log_ratio| ~= 0.00966`

The small mean drift becomes pathological near bounded-action saturation.
Using the measured `0.005267` drift, the boundary diagnostic produced:

- margin `0.2` from upper bound: `log_ratio ~= -0.266`
- margin `0.05`: `log_ratio ~= -4.61`
- margin `0.01`: `log_ratio ~= -207.95`
- margin `0.002`: `log_ratio ~= -3311`

This explains how a small fraction of saturated transitions can create very
large pre-update KL even when ACT parameters have not changed.

## Related PPO protections that must remain

1. Rollout `old_log_prob` is canonicalized from the final stored bounded
   action.
2. Pre-update old/new log-prob consistency is checked before any
   `optimizer.step()`.
3. Target KL is `0.02`, checked at complete PPO epoch boundaries.
4. Do not bypass a failed pre-update consistency check by widening its
   tolerance.

Expected pre-update behavior after enforcing this invariant:

- KL approximately zero
- clip fraction zero
- maximum absolute log-ratio approximately zero

## State 1

State 1 remains the no-target stationary state. Exploration only supplies PPO
learning signal; the behavior objective remains to stay still without a
target.
