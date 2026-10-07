"""Run a small CLEVR eval while forcing all HiDESC layers to one expert."""

import argparse
import json
import os
import types

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.CoIN.coin_utils import get_model_name_from_path
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def _as_list(tensor):
    if tensor is None:
        return None
    return tensor.detach().float().cpu().tolist()


def _normalize_questions(questions, limit):
    if limit is None:
        return questions
    return questions[: max(0, int(limit))]


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        num_task=args.num_task,
        text_tower=args.text_tower,
        routing_config_path=args.routing_config_path,
        stage1_band_schedule_path=args.stage1_band_schedule_path,
    )
    if image_processor is None:
        raise RuntimeError(f"Failed to initialize image processor: {model_path}")

    force_expert = int(args.force_expert)
    route_records = []
    current_sample = None

    def forced_route_builder(self, active_experts, task_scores, image_summary, text_summary):
        nonlocal current_sample
        if force_expert < 0 or force_expert >= int(active_experts):
            raise ValueError(
                f"force_expert={force_expert} is outside active_experts={active_experts}"
            )
        weights = torch.zeros(
            int(active_experts),
            device=task_scores.device,
            dtype=torch.float32,
        )
        weights[force_expert] = 1.0
        role_scores = self._score_roles(
            active_experts,
            task_scores,
            image_summary,
            text_summary,
            image_summary.device,
        )
        if current_sample is not None:
            route_records.append(
                {
                    "question_id": current_sample["question_id"],
                    "forced_expert": force_expert,
                    "candidate_experts": [force_expert],
                    "active_experts": int(active_experts),
                    "role_scores": _as_list(role_scores),
                    "task_scores": _as_list(task_scores),
                    "avg_expert_activation": _as_list(weights),
                    "role_member_tasks": {
                        str(role_id): self._role_member_tasks(role_id, active_experts)
                        for role_id in range(
                            min(
                                self._get_active_role_count(),
                                self._max_supported_role_slots(),
                            )
                        )
                    },
                }
            )
        return {
            "early_basis": weights,
            "middle_basis": weights,
            "late_basis": weights,
            "per_layer": [weights for _ in range(len(self.model.layers))],
            "candidate_experts": [force_expert],
            "role_scores": role_scores,
        }

    model._build_progressive_route_plan = types.MethodType(forced_route_builder, model)
    model.eval()

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as handle:
        questions = _normalize_questions(json.load(handle), args.limit)

    answers_file = os.path.abspath(os.path.expanduser(args.answers_file))
    route_file = os.path.abspath(os.path.expanduser(args.route_file))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.dirname(route_file), exist_ok=True)

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(questions, desc=f"forced expert {force_expert}"):
            current_sample = line
            question_id = line["question_id"]
            image_file = line["image"]
            raw_question = line["text"]

            question = raw_question
            if model.config.mm_use_im_start_end:
                question = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + question
                )
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            image_path = os.path.join(args.image_folder, image_file)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = image_processor.preprocess(
                    image,
                    return_tensors="pt",
                )["pixel_values"][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            output_text = tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True,
            )[0].strip()
            if output_text.endswith(stop_str):
                output_text = output_text[: -len(stop_str)].strip()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "prompt": raw_question,
                        "text": output_text,
                        "answer_id": shortuuid.uuid(),
                        "model_id": f"{model_name}-forced-expert{force_expert}",
                        "metadata": {"forced_expert": force_expert},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()
            current_sample = None

    if len(route_records) != len(questions):
        raise RuntimeError(
            f"Expected {len(questions)} route records, got {len(route_records)}"
        )
    with open(route_file, "w", encoding="utf-8") as handle:
        for record in route_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--route-file", required=True)
    parser.add_argument("--text-tower", required=True)
    parser.add_argument("--num-task", type=int, required=True)
    parser.add_argument("--routing-config-path", required=True)
    parser.add_argument("--stage1-band-schedule-path", required=True)
    parser.add_argument("--force-expert", type=int, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    main(parser.parse_args())
