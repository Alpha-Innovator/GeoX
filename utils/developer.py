import os
import warnings
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from models.geox import GeoXLlamaForCausalLM
from configs.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_SEP_TOKEN

from torchvision import transforms
from PIL import Image
from data.processor import ImageProcessor


def load_pretrained_model(model_path, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}


    kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    kwargs['device_map']="cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = GeoXLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_use_sep_token = getattr(model.config, "mm_use_sep_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_sep_token:
        tokenizer.add_tokens([DEFAULT_SEP_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)

    if device_map != 'auto':
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = ImageProcessor(mode='val')

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
