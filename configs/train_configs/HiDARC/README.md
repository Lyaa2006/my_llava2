# HiDARC training configuration

The four `final.json` profiles below record run resources and optimization
settings. They follow the repository convention used by the other methods:
`rank` is the LoRA rank and `expert_num` is the number of task experts. The
launchers derive `lora_alpha = 2 * rank`.

| Profile | Reference run | Model | Protocol |
| --- | --- | --- | --- |
| `LLaVA/UCIT/train/final.json` | 2026-08-27 `hidesc_new_strategy_full_20260827_103909` | LLaVA | UCIT, 6 tasks |
| `LLaVA/MLLM-DCL/train/final.json` | 2026-08-27 `hidesc_new_strategy_full_20260827_103909` | LLaVA | DCL, 5 tasks |
| `InternVL/UCIT/train/final.json` | 2026-09-18 `ucit_gpu7_rolebank829_last_eval_20260918` | InternVL | UCIT, 6 tasks |
| `InternVL/MLLM-DCL/train/final.json` | 2026-09-25 native 3-role matrix | InternVL | DCL, 5 tasks |

The task-specific `task*.json` files retain per-task `cur_task`, epoch,
checkpoint, and description-cache paths. Use the final profiles as the
canonical hyperparameter reference when generating a new run-local task
configuration.

The final FFT anchor extraction and fixed stage-band policy are implemented in
`LLaVA/HiDARC/llava/model/hidarc_final.py` and
`InternVL/HiDARC/llava/model/hidarc_final.py`; those fixed values are not in
JSON. The four `final.json` profiles independently retain their role-bank
assignment parameters; evaluation task JSON files retain the corresponding
experiment-specific routing activation parameters. The anchor is the online
sample-weighted mean of the same per-example descriptor used offline, so it is
independent of batch partition. A later task rejects a prior checkpoint unless
its persisted anchor/role profile matches exactly.
