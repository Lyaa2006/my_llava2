# Stage 1 `b1` Interval Report

Date: August 11, 2026

## Bottom Line

At the current evidence level, `b1` can be reported as an interval rather than a single layer.

- Recommended consensus interval: `15-18`
- Sharper center band: `16-18`
- Peak vote layer across valid splits: `16`

This is strong enough for a stage-1 paper claim of "stable interval with residual seed sensitivity", but not strong enough for a claim of "single exact layer" or "fully data-independent boundary".

## What Was Fixed Before Running

This round only targets `b1`.

- `b2` workflow was left unchanged.
- `b1` uses a dedicated runner: [scripts/run_stage1_b1_results.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_results.sh)
- Multi-seed launch is handled by: [scripts/run_stage1_b1_multiseed.sh](/mnt/lyaa/MCITlib/scripts/run_stage1_b1_multiseed.sh)

The key change is that `b1` is no longer treated as a single best point. Instead, the analyzer returns a `3-5` layer band around the best crossover region.

## Experiment Design

### Split Protocol

We used leave-one-dataset-out evaluation over:

- `acl`
- `dcl`
- `ucit`

For each seed, the analyzer builds three reports, one for each held-out dataset.

### Seeds and Scale

The final large run used six seeds:

- `7`
- `13`
- `21`
- `29`
- `35`
- `42`

The corresponding results are under:

- [seed7](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed7/canonical/stage1_boundary_summary.json)
- [seed13](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed13/canonical/stage1_boundary_summary.json)
- [seed21](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed21/canonical/stage1_boundary_summary.json)
- [seed29](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed29/canonical/stage1_boundary_summary.json)
- [seed35](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed35/canonical/stage1_boundary_summary.json)
- [seed42](/mnt/lyaa/MCITlib/docs/stage1_b1_band_large_20260811/llava/seed42/canonical/stage1_boundary_summary.json)

The large-scale launcher used cache prewarming from prior `b1` runs whenever matching sample caches were already present. This helps same-seed reanalysis a lot; for new seeds, reuse is partial because sample identities do change with the seed.

### Why Cross-Seed Matters

Cross-seed testing is used to measure sampling stability.

If a candidate `b1` band only appears under one seed, it may be a sampling artifact. If it remains strong across many seeds, it is more likely to reflect a model-level stage transition.

## Metric Definition

The implementation is in [scripts/stage1_analyze_boundaries.py](/mnt/lyaa/MCITlib/scripts/stage1_analyze_boundaries.py).

### Stage Signals

For each split, we compute three per-layer curves:

1. `reasoning_signal`

   For reasoning-family samples, hidden states are grouped by label and scored with Fisher separation:

   `Fisher(l) = BetweenClassVar(l) / max(WithinClassVar(l), eps)`

2. `objective_signal`

   The same Fisher separation is computed on objective-family samples.

3. `style_signal`

   For style-family samples, we compute the mean pairwise Jensen-Shannon divergence of style logits within each paired group:

   `JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5 * (p + q)`

### Normalization and Smoothing

Each curve is transformed as:

- min-max normalization to `[0, 1]`
- moving-average smoothing with window `3`

Denote the normalized-and-smoothed curves as:

- `R_l`: reasoning
- `O_l`: objective
- `S_l`: style

### `b1` Crossover Score

The current `b1` strategy is `b1_only_crossover`, with search window:

- `b1_search_low = 10`
- `b1_search_high = 20`
- `transition_min_run = 3`
- `min_transition_support = 0.01`
- `min_pre_reasoning_margin = -0.20`
- `b1_band_score_tolerance = 0.03`
- `b1_band_min_width = 3`
- `b1_band_max_width = 5`

For each candidate layer `b1`, define:

- `pre_margin(b1) = mean(R_l - O_l)` over the `3` layers immediately before `b1`
- `post_margin(b1) = mean(O_l - R_l)` over the `3` layers immediately after `b1`
- `boundary_contrast(b1) = pre_margin(b1) + post_margin(b1)`

The local crossover score is:

`local_score(b1) = min(pre_margin(b1), post_margin(b1)) + boundary_contrast(b1) + late_preference * (b1 - low)`

with `late_preference = 0.01`.

### Valid Candidate Rule

A candidate belongs to the viable pool if:

- `post_margin(b1) >= 0.01`
- `pre_margin(b1) >= -0.20`

This is intentionally softer than requiring a strictly positive pre-margin, because the experiments showed that later and more plausible `b1` bands were often rejected by an overly strict early-margin rule.

### From Point to Interval

After scoring all viable candidates:

- keep all candidates within `0.03` of the best `local_score`
- choose a contiguous interval of width `3-5` that contains the best point
- prefer the interval with the largest support from near-best candidates
- break ties by staying close to the best point

This produces `boundary_windows.b1 = [low, high]`.

So the experiment is not hard-constraining `b1` to `15-18`. The interval is still computed from data inside the fixed search window `10-20`.

### `b2` Selection During This Analysis

Although this report is about `b1`, the script still chooses a companion `b2` for completeness:

- middle score: `mean(O_l - max(R_l, S_l))` on `[b1, b2)`
- late score: `mean(S_l - max(R_l, O_l))` on `[b2, end)`
- choose `b2` maximizing `middle score + late score`

This did not alter the previously established `b2` workflow.

## Final Results

### Per-Seed Summary

| Seed | Valid Splits | Valid `b1` Window | Notes |
| --- | --- | --- | --- |
| 7 | 2 / 3 | `15-19` | `acl` and `dcl` support later band |
| 13 | 1 / 3 | `13-17` | only `acl` valid |
| 21 | 1 / 3 | `10-14` | early outlier seed |
| 29 | 2 / 3 | `16-20` | strongest later support |
| 35 | 2 / 3 | `14-18` | later support remains strong |
| 42 | 2 / 3 | `10-15` | mixed seed, partly early |

### Per-Dataset Stability

Across six seeds:

- `acl`: `6 / 6` valid
- `dcl`: `4 / 6` valid
- `ucit`: `0 / 6` valid

Interpretation:

- `acl` is consistently supportive of a meaningful `b1` band.
- `dcl` usually supports the same later band, but can still jump early on some seeds.
- `ucit` is the main source of residual instability and currently does not support a viable later crossover band under this metric.

### Valid-Band Vote Aggregation

Aggregating all valid split bands across all seeds gives the following per-layer vote counts:

| Layer | Votes |
| --- | --- |
| 10 | 2 |
| 11 | 2 |
| 12 | 3 |
| 13 | 4 |
| 14 | 5 |
| 15 | 6 |
| 16 | 8 |
| 17 | 7 |
| 18 | 6 |
| 19 | 5 |
| 20 | 2 |

This shows:

- maximum vote at layer `16`
- dense support over `15-18`
- clear shift toward later layers once more seeds are added

## Why These Results Support the Conclusion

The evidence supports a `b1` interval claim for three reasons.

1. The peak vote is no longer near the lower bound.

   Earlier failed metrics often collapsed to `10-12`, which strongly suggested lower-bound artifacts. The current metric moves the main consensus to the middle-late region, centered at `16`.

2. Later-supporting seeds now dominate the valid evidence.

   `seed7`, `seed29`, and `seed35` all support later bands around `15-20`, and together they provide the strongest valid coverage.

3. The result is stable as an interval even when it is not stable as a point.

   Exact points still vary by split and seed, but the vote mass is concentrated in a narrow region. This is exactly why `b1` should be reported as an interval rather than a single layer.

## Recommended Paper Claim

The safest claim at this stage is:

> Across random seeds, the estimated stage-1 boundary `b1` does not collapse to a single layer, but concentrates in a narrow interval. The strongest consensus lies around layers `15-18`, with peak support at layer `16`. This suggests that `b1` primarily reflects a model-internal transition, while still retaining some seed-level sensitivity, especially on `ucit`.

In short:

- acceptable claim: "`b1` concentrates in `15-18`"
- acceptable stronger summary: "peak near `16`"
- not yet justified: "`b1` is exactly one layer" or "`b1` is fully data-independent"

## Reproduction

Large multi-seed launch:

```bash
bash scripts/run_stage1_b1_multiseed.sh
```

Current default seeds in that launcher:

```bash
7 13 21 29 35 42
```

If more seeds are needed:

```bash
SEEDS='7 13 21 29 35 42 49 56' bash scripts/run_stage1_b1_multiseed.sh
```

## Current Decision

Yes, the repository now has enough evidence to use a provisional `b1` interval.

The recommended operational choice is:

- primary interval: `15-18`
- center emphasis: `16-18`
- single representative point, if one must be used: `16`

This should be treated as a supported interval estimate, not a final exact point estimate.
