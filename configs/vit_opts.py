import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class GeoViTArguments:
    
    # Model Arguments
    pretrained_ckpt: str = field(
        default='./pretrained',
        metadata={"help": "Path to pretrained checkpoint."}
    )
    input_size: int = field(
        default=224,
        metadata={"help": "Images input size."}
    )
    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Masking ratio (percentage of removed patches)."}
    )
    norm_pix_loss: bool = field(
        default=False,
        metadata={"help": "Use normalized pixels as targets for computing loss."}
    )

    # Data Arguments
    data_path: str = field(
        default='./data',
        metadata={"help": "Dataset path."}
    )
    num_workers: int = field(
        default=10,
        metadata={"help": "Number of data loading workers."}
    )
    pin_mem: bool = field(
        default=True,
        metadata={"help": "Pin CPU memory in DataLoader for more efficient transfer to GPU."}
    )

    # Training Arguments:
    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU."}
    )
    epochs: int = field(
        default=400,
        metadata={"help": "Number of epochs for training."}
    )
    accum_iter: int = field(
        default=1,
        metadata={"help": "Gradient accumulation iterations."}
    )
    weight_decay: float = field(
        default=0.05,
        metadata={"help": "Weight decay."}
    )
    lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate."}
    )
    blr: float = field(
        default=1e-3,
        metadata={"help": "Base learning rate."}
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "Lower learning rate bound."}
    )
    warmup_epochs: int = field(
        default=40,
        metadata={"help": "Warmup epochs."}
    )
    output_dir: str = field(
        default='./pretrained',
        metadata={"help": "Path to save output."}
    )
    log_dir: str = field(
        default='./pretrained',
        metadata={"help": "Path for tensorboard logs."}
    )
    device: str = field(
        default='cuda',
        metadata={"help": "Device for training/testing."}
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed."}
    )
    resume: str = field(
        default='',
        metadata={"help": "Path to resume checkpoint."}
    )
    start_epoch: int = field(
        default=0,
        metadata={"help": "Start epoch."}
    )
    world_size: int = field(
        default=1,
        metadata={"help": "Number of distributed processes."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training."}
    )
    dist_on_itp: bool = field(
        default=False,
        metadata={"help": "Use distributed training on itp."}
    )
    dist_url: str = field(
        default='env://',
        metadata={"help": "URL for setting up distributed training."}
    )

def parse_arguments(argument_class):
    # Filter out distributed training arguments like --local-rank
    argv = [arg for arg in sys.argv[1:] if not arg.startswith("--local-rank")]

    # Parse the filtered arguments
    parser = HfArgumentParser(argument_class)
    args = parser.parse_args_into_dataclasses(args=argv)[0]
    
    return args