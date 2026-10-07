# HiDARC training configuration

The four `final.json` profiles below record the hyperparameters used by the
completed HiDARC runs. They follow the repository convention used by the
other methods: `rank` is the LoRA rank and `expert_num` is the number of task
experts. The launchers derive `lora_alpha = 2 * rank`.

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

Each protocol uses one eval JSON per task under its `eval/` directory. Every
task file contains the checkpoint/evaluator fields, the routing file actually
loaded by the evaluator, and an inline `activation` object with the active
spectral/text/role/layer-routing hyperparameters.

Routing is kept separately under `configs/routing_configs/HiDARC/` because
UCIT and DCL use different role-bank and layer-routing policies. The final
run references are:

- LLaVA UCIT: `ucit_role_new_strategy_eval_sharp_canonical_llava_20260827.json`
- LLaVA DCL: `dcl_role_new_partition_eval_late_role_prototype_only.json`
- InternVL UCIT: `ucit_role_internvl_4role_eval_v2_rebuild50_20260908.json`
- InternVL DCL: `dcl_probe_activation_balanced_role3_rebuild_20260922.json`
