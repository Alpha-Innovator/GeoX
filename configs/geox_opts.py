from dataclasses import dataclass, field
from typing import Optional
from configs.opts import ModelArguments as BaseModelArguments, DataArguments as BaseDataArguments
import transformers

@dataclass
class ModelArguments(BaseModelArguments):
    """
    Arguments related to model configuration, including the model path, version, and various options
    for configuring multimodal features. Inherits from the base ModelArguments.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the cached training data and models."}
    )
    version: Optional[str] = field(
        default="v0",
        metadata={"help": "Version of the model."}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone model during fine-tuning."}
    )
    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to tune the multimodal MLP adapter."}
    )
    vision_tower: Optional[str] = field(
        default=None,
        metadata={"help": "Name or path to the vision model used for multimodal training."}
    )
    mm_vision_select_layer: Optional[int] = field(
        default=-1,
        metadata={"help": "Select which layer of the vision tower to use. Default is the last layer."}
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained multimodal MLP adapter, if available."}
    )
    mm_projector_type: Optional[str] = field(
        default='linear',
        metadata={"help": "Type of projector to use for multimodal MLP adapter. Options include 'linear'."}
    )
    mm_use_im_start_end: bool = field(
        default=False,
        metadata={"help": "Whether to use image start and end tokens for multimodal inputs."}
    )
    mm_use_im_patch_token: bool = field(
        default=True,
        metadata={"help": "Whether to use patch tokens for image representation in multimodal setup."}
    )
    mm_use_sep_token: bool = field(
        default=False,
        metadata={"help": "Whether to use a separate token to divide modalities."}
    )
    mm_patch_merge_type: Optional[str] = field(
        default='flat',
        metadata={"help": "How to merge image patches. Default is 'flat'."}
    )
    mm_vision_select_feature: Optional[str] = field(
        default="patch",
        metadata={"help": "Select which feature to use from vision model. Default is 'patch'."}
    )
    gsformer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained processor (e.g., tokenizer or feature extractor)."}
    )


@dataclass
class DataArguments(BaseDataArguments):
    """
    Arguments related to data input for training and evaluation, including paths and processing options.
    Inherits from the base DataArguments.
    """
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "Whether to enable lazy preprocessing of the dataset."}
    )
    is_multimodal: bool = field(
        default=False,
        metadata={"help": "Specify whether the dataset is multimodal (e.g., includes images)."}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing image data, if applicable."}
    )
    image_aspect_ratio: str = field(
        default='square',
        metadata={"help": "Aspect ratio for images in the dataset. Options: 'square', etc."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments specific to the training configuration, including optimization, model saving, and quantization.
    """
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "Optimization algorithm to use for training."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."}
    )
    freeze_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the multimodal MLP adapter during training."}
    )
    mpt_attn_impl: Optional[str] = field(
        default="triton",
        metadata={"help": "Type of attention implementation to use, e.g., 'triton'."}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum input sequence length. Longer sequences will be truncated."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to apply double quantization for model compression."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type to use. Options: 'fp4', 'nf4'."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "Number of bits to use for model quantization."}
    )
    lora_enable: bool = field(
        default=False,
        metadata={"help": "Whether to enable LoRA (Low-Rank Adaptation) during training."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Rank of the LoRA adaptation."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA adaptation."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout rate for LoRA."}
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "Path to a pre-trained LoRA weight, if applicable."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias term for LoRA. Options: 'none', etc."}
    )
    mm_projector_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for the multimodal projector, if applicable."}
    )
    group_by_modality_length: bool = field(
        default=False,
        metadata={"help": "Whether to group inputs by modality length for more efficient batching."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility of results."}
    )
