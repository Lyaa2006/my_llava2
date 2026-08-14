# InternVL MoELoRA + HiDESC Train

This ablation path keeps only the HiDESC training losses:

```text
total = standard_ce_weight * CE
      + align_loss_weight * B1_align
      + description_focus_weight * B2_focus
      + description_energy_weight * B2_energy
```

The InternVL defaults are:

```text
B1 = [12, 14]
B2 = [27, 29]
```

Prepare the B2-high reference states first:

```bash
bash scripts/MCITlib/Train/extract_description_cache.sh \
  <model.json> <data.json> <train.json>
```

Then run Task 1 or Task N:

```bash
bash scripts/MCITlib/Train/Task1_hidesc.sh <model.json> <data.json> <train.json>
bash scripts/MCITlib/Train/Taskn_hidesc.sh <model.json> <data.json> <train.json>
```

The cache extractor uses the base InternVL model by default. If `previous_model`
is present in the train config, it loads that previous MoELoRA checkpoint before
extracting the cache. No training command is run by this preparation step.
