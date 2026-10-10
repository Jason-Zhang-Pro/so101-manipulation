# Experimental Evidence

## Data and evaluation

The principal black-tabletop dataset contains 150 demonstrations at 15 grid
centres, nominally 10 per position. The reported ACT configuration uses dual RGB
cameras, ImageNet ResNet18, batch size 8, chunk size 50, KL weight 0.5 and temporal
ensemble coefficient 0.005. The identified baseline checkpoint is step 100000.
Its inspected configuration has image augmentation disabled. Other augmented
models are mentioned in the log; their results must not be attributed to this
checkpoint without an experiment manifest.

Full seen-centre evaluation reports 128 successes in 150 trials. P1/R1C2 and
P6/R2C2 each achieved 6/10. Labels are zero-based row-major:

```
P0  P1  P2  P3  P4
P5  P6  P7  P8  P9
P10 P11 P12 P13 P14
```

Unseen-position success was reported as 25%; trial count and exact test-point
coordinates are not supplied in the slides. Lighting evaluations used 30 trials
per condition (2 per cell), which is insufficient for strong per-cell claims.

## Interpretation limits

- Training prediction errors are measured in action space, not millimetres.
- Prediction variance across demonstrations includes input-state variation.
- Similar globally pooled camera features do not prove harmful camera fusion.
- Recovered labels for occluded replacement episodes need independent checking.
- Selecting the two worst cells after evaluation requires care with significance
  tests; an unadjusted pooled p-value is exploratory rather than confirmatory.
- Earlier hypotheses about latent collapse and averaged actions remain hypotheses.

## Simulation performance

Week 10 reports four-world RTF improvements from 0.614 to 0.971 with GUI and
0.696 to 0.977 without GUI. The headless run reported approximately 32% lower CPU
usage and 18% lower proportional-set-size memory. These measurements excluded
ACT inference and PPO updates. The benchmark harness and optimized parallel
runtime were not located in the current source snapshot; only their reported
results are included here.

## Next controlled experiments

1. Save per-trial video, qpos, commanded action and measured grasp failure direction.
2. Compare image augmentation using the same demonstrations and evaluation setup.
3. Test fixed-only versus dual-camera models with matched training settings.
4. Validate unseen-position performance after a defined ACT-to-RL refinement stage.
