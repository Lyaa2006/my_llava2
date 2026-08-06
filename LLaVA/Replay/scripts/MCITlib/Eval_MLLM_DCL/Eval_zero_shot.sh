# #!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/Replay/LLaVA/MLLM-DCL/eval/zero_shot.json
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/Replay/LLaVA/MLLM-DCL/eval/zero_shot.json
bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/Replay/LLaVA/MLLM-DCL/eval/zero_shot.json
bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/Replay/LLaVA/MLLM-DCL/eval/zero_shot.json
bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $HARD_PATH/configs/train_configs/Replay/LLaVA/MLLM-DCL/eval/zero_shot.json