# L_focus Logic-Oriented Experiment Report

## What This Script Tries To Test

This report does not assume that every task must show uniform degradation after training.
Instead, it checks whether larger template-token drift tends to coincide with weaker cross-template semantic stability,
which is the specific failure mode that `L_focus` is meant to suppress.

## Analysis Principles

- Do not require every task to show degradation; early tasks may sharpen representations.
- Treat harmful template drift as a coupling pattern: template drift should align with semantic instability.
- Use late-task behavior as stronger evidence than early-task behavior in continual learning.
- Do not overclaim causality from Phase A alone; ablation is required for causal support.

## Group Summary

### All Tasks

- task ids: [1, 2, 3, 4, 5, 6]
- harmful template-drift support rate: 0.500
- mean alignment drop: 0.0112
- mean retrieval drop: -0.0032
- mean template/alignment corr: 0.0735
- mean template/retrieval corr: -0.0083

### Early Tasks

- task ids: [1, 2]
- harmful template-drift support rate: 0.000
- mean alignment drop: -0.0042
- mean retrieval drop: -0.0033
- mean template/alignment corr: -0.0666
- mean template/retrieval corr: 0.1639

### Late Tasks

- task ids: [3, 4, 5, 6]
- harmful template-drift support rate: 0.750
- mean alignment drop: 0.0189
- mean retrieval drop: -0.0031
- mean template/alignment corr: 0.1435
- mean template/retrieval corr: -0.0944

## Per-Task Logic

### Task 1 / ImageNet-R

- conclusion: `mixed_signal`
- harmful channels: []
- alignment drop mean: -0.0054
- retrieval drop mean: -0.0066
- margin drop mean: 0.0064
- template/alignment corr: -0.1280
- template/retrieval corr: 0.0692
- template drift mean: 15.1254
- content drift mean: 33.4678

### Task 2 / ArxivQA

- conclusion: `weak_or_null`
- harmful channels: []
- alignment drop mean: -0.0029
- retrieval drop mean: 0.0000
- margin drop mean: -0.0011
- template/alignment corr: -0.0052
- template/retrieval corr: 0.2587
- template drift mean: 4.3881
- content drift mean: 5.7709

### Task 3 / VizWiz

- conclusion: `partial_support`
- harmful channels: ['alignment']
- alignment drop mean: 0.0318
- retrieval drop mean: -0.0038
- margin drop mean: 0.0211
- template/alignment corr: 0.0981
- template/retrieval corr: -0.1874
- template drift mean: 13.8117
- content drift mean: 26.0538

### Task 4 / IconQA

- conclusion: `strong_support`
- harmful channels: ['alignment', 'margin']
- alignment drop mean: 0.0253
- retrieval drop mean: -0.0009
- margin drop mean: 0.0152
- template/alignment corr: 0.4497
- template/retrieval corr: -0.0452
- template drift mean: 8.4055
- content drift mean: 12.3355

### Task 5 / CLEVR

- conclusion: `partial_support`
- harmful channels: ['margin']
- alignment drop mean: 0.0205
- retrieval drop mean: 0.0000
- margin drop mean: 0.0119
- template/alignment corr: 0.0121
- template/retrieval corr: -0.1984
- template drift mean: 8.5190
- content drift mean: 10.1411

### Task 6 / Flickr30k

- conclusion: `weak_or_null`
- harmful channels: []
- alignment drop mean: -0.0019
- retrieval drop mean: -0.0076
- margin drop mean: -0.0029
- template/alignment corr: 0.0140
- template/retrieval corr: 0.0533
- template drift mean: 13.2387
- content drift mean: 17.1863

## Overall Conclusion

- overall conclusion: `mixed_but_meaningful_support`
- This Phase A report is meant to motivate `L_focus`, not to prove its causal necessity by itself.
