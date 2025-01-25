import torch
import torch.nn as nn

from transformers import ViTMAEModel, AutoImageProcessor, ViTMAEConfig

import json

from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


from lavis.modules.blip2 import (
    Blip2Base,
)

from models.gsformer import GSFormer

from functools import partial
import timm.models.vision_transformer
from utils.pos_embed import interpolate_pos_embed

from torchvision import transforms
from PIL import Image
from data.processor import ImageProcessor
from models.vit_encoder import VTransformer
import yaml
CONFIG = 'configs/param.yaml'


def frozen_params(model):
    with open(CONFIG, 'r') as f:
        frozen_param_names = yaml.safe_load(f)

    for name, param in model.named_parameters():
        if name in frozen_param_names:
            param.requires_grad = False
        else:
            pass

class GeoVisionTower(Blip2Base):

    def __init__(self, vision_tower, args, delay_load=False, num_query_token=8, cross_attention_freq=2, feature_dim=768):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
            
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # self.processor_path = args.processor_path
        if hasattr(args, 'processor_path'):
            self.processor_path = args.processor_path
            
 
        self.gsformer = GSFormer(num_query_token, feature_dim, cross_attention_freq, finetune=True)

        self.gsformer.mmtransformer.bert.embeddings.word_embeddings = None
        self.gsformer.mmtransformer.bert.embeddings.position_embeddings = None
        for layer in self.gsformer.mmtransformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.gsformer.mmtransformer.cls = None
        if hasattr(args, 'gsformer_path') and args.gsformer_path:
            gsformer_path = args.gsformer_path
        else:
            gsformer_path = None
        self.load_model(gsformer_path)

    
    def load_model(self, gsformer_path=None, device_map=None):

        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load Vision Encoder
        self.image_processor = ImageProcessor()
        self.vision_encoder = VTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        checkpoint = torch.load(self.vision_tower_name, map_location='cpu')
        checkpoint_model = checkpoint['model']

        state_dict = self.vision_encoder.state_dict()

        # Remove mismatched keys
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # Interpolate position embeddings and load state dict
        interpolate_pos_embed(self.vision_encoder, checkpoint_model)
        msg = self.vision_encoder.load_state_dict(checkpoint_model, strict=False)
        print("The keys are not loaded by Vision Encoder:", msg.missing_keys)

        if CONFIG is not None:
            frozen_params(self.vision_encoder)
            self.vision_encoder.eval()

        # Load GSFormer checkpoint if provided
        if gsformer_path:
            new_state_dict = {}
            gsformer_checkpoint = torch.load(gsformer_path, map_location='cpu')
            
            for key, value in gsformer_checkpoint['model'].items():
                key = key.split("gsformer.")[-1]
                new_state_dict[key] = value

            missing_keys, unexpected_keys = self.gsformer.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys when loading GSFormer: {missing_keys}")

        
            if CONFIG is not None:
                frozen_params(self.gsformer)
        self.is_loaded = True


    def forward(self, images):
        with torch.no_grad():
            if type(images) is list:
                image_embeds = []
                for image in images:
                    image_feature = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_embeds.append(image_feature)
            else:
                image_embeds = self.vision_encoder(images).contiguous()

        query_output = self.gsformer(image_embeds)
        return query_output

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



