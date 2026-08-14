import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pandas as pd
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.utils import filter_samples_with_existing_images
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from llava.eval.CoIN.coin_utils import get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _literal_or_str(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return repr(value)


def _question_row_id(question_id):
    if not isinstance(question_id, str):
        return None
    match = re.search(r"(\d+)$", question_id)
    if match is None:
        return None
    return int(match.group(1))


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        filtered_questions, skipped_samples = filter_samples_with_existing_images(questions, image_folder)
        if skipped_samples:
            print(
                f"Skipping {len(skipped_samples)} bad image samples out of {len(questions)} "
                f"for image_folder={image_folder}"
            )
            for item in skipped_samples[:5]:
                print(f"  - {item['image']} ({item['reason']})")
        self.questions = filtered_questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, num_task=args.num_task, text_tower=args.text_tower)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    questions = data_loader.dataset.questions
    row_lookup = {}
    template_path = 'llava_v1.5_7b_MathVista_MINI.xlsx'
    if os.path.exists(template_path):
        excel_template = pd.read_excel(template_path)
        template_rows = [_question_row_id(line.get("question_id")) for line in questions]
        if template_rows and all(row is not None and 0 <= row < len(excel_template) for row in template_rows):
            excel_ori = excel_template.iloc[template_rows].copy().reset_index(drop=True)
            row_lookup = {
                line["question_id"]: pos
                for pos, line in enumerate(questions)
                if "question_id" in line
            }
        else:
            excel_ori = excel_template
    else:
        records = []
        for line in questions:
            record = dict(line)
            record['index'] = line.get('question_id', len(records))
            record['question'] = line.get('question', line.get('text', ''))
            record.setdefault('prediction', '')
            record.setdefault('task', 'Math')
            record['skills'] = _literal_or_str(line.get('skills', ['Math']), "['Math']")
            record['choices'] = _literal_or_str(line.get('choices', []), '[]')
            record.setdefault('answer_option', line.get('answer_option', ''))
            record.setdefault('question_type', line.get('question_type', 'free_form'))
            answer = line.get('answer')
            if 'answer_type' in line:
                record['answer_type'] = line['answer_type']
            elif isinstance(answer, int) and not isinstance(answer, bool):
                record['answer_type'] = 'integer'
            elif isinstance(answer, float):
                record['answer_type'] = 'float'
            else:
                record['answer_type'] = 'text'
            records.append(record)
        excel_ori = pd.DataFrame(records)
        row_lookup = {
            line["question_id"]: pos
            for pos, line in enumerate(questions)
            if "question_id" in line
        }
    excel = copy.deepcopy(excel_ori)
    for row_idx, ((input_ids, image_tensor), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions))):
        idx = line["question_id"]
        cur_prompt = line["text"]
        id_excel = row_lookup.get(idx, row_idx)
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        conv = conv_templates[args.conv_mode].copy()
        stop_str =conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str] # [</s>]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                    use_cache=True)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[skip-oom] question_id={idx} image={line.get('image')} reason={exc}", flush=True)
            torch.cuda.empty_cache()
            continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[skip-oom] question_id={idx} image={line.get('image')} reason={exc}", flush=True)
            torch.cuda.empty_cache()
            continue

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        excel.at[id_excel, 'prediction'] = outputs
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        
    ans_file.close()

    excel.to_excel(args.output_xlsx, index=False)
    print('Excel saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_xlsx", type=str, default="output.xlsx")
    parser.add_argument("--text-tower", type=str)
    parser.add_argument("--num-task", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
