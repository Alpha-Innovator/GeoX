import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from configs.conversation import conv_templates, SeparatorStyle
from utils.developer import load_pretrained_model
from utils.mm_utils import disable_torch_init, tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math




def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, device_map="cuda")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
    
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images(image, image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i in range(len(outputs)):
            outputs[i] = outputs[i].strip()
            if outputs[i].endswith(stop_str):
                outputs[i] = outputs[i][:-len(stop_str)]
            outputs[i] = outputs[i].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="geo_v1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    args = parser.parse_args()

    main(args)
