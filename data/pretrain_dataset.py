import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
import torchvision
import utils.misc as misc

from datasets import load_dataset
import transformers
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def load_and_prepare_dataset(data_args, model_args, training_args):
    # Set up data files and dataset arguments
    data_files = {}
    dataset_args = {}

    if data_args.train_files is not None:
        print(data_args.train_files)
        data_files["train"] = data_args.train_files

    if data_args.validation_files is not None:
        data_files["validation"] = data_args.validation_files

    extension = (
        data_files["train"][0].split(".")[-1]
        if data_files.get("train") is not None
        else data_args.validation_files.split(".")[-1]
    )

    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

    # Load datasets
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        streaming=data_args.streaming,
        cache_dir=os.path.join(training_args.output_dir, 'dataset_cache'),
        use_auth_token=True if model_args.use_auth_token else None,
        **dataset_args,
    )

    # Shuffle datasets if streaming
    if data_args.streaming:
        raw_datasets = raw_datasets.shuffle(seed=training_args.seed, buffer_size=1000000)

    # Split train set for validation if no validation files are provided
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )

    return raw_datasets


def tokenize_and_group(raw_datasets, tokenizer, data_args, training_args, text_column_name):
    # Tokenization logger setup
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # Define tokenization function
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer([item for item in examples[text_column_name]])
        return output

    # Tokenize datasets
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                batch_size=60000,
            )

    # Set block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            transformers.utils.logging.warning(
                "The chosen tokenizer supports a model_max_length that is longer than the default block_size value"
                " of 1024. If you would like to use a longer block_size up to tokenizer.model_max_length you can"
                " override this default with --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            transformers.utils.logging.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Group texts into blocks
    def group_texts(examples):
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        for ids in examples['input_ids']:
            truncated_ids = ids[:block_size]
            padding_length = block_size - len(truncated_ids)

            batch['input_ids'].append(truncated_ids + [tokenizer.pad_token_id] * padding_length)
            batch['attention_mask'].append([1] * len(truncated_ids) + [0] * padding_length)
            batch['labels'].append(truncated_ids + [tokenizer.pad_token_id] * padding_length)

        return batch

    # Map group_texts function to the tokenized datasets
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                lambda examples: group_texts(examples),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping and padding texts in chunks of {block_size}",
                batch_size=40000,
            )
        else:
            lm_datasets = tokenized_datasets.map(
                lambda examples: group_texts(examples),
                batched=True,
                batch_size=60000,
            )

    return lm_datasets


class GeoViTDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=self.transform)

    def __len__(self):
        return len(self.dataset_train)

    def __getitem__(self, index):
        return self.dataset_train[index]

    def get_data_loader(self):
        sampler_train = self._create_sampler()
        return DataLoader(
            self.dataset_train,
            sampler=sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

    def _create_sampler(self):
        if self.args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            return DistributedSampler(
                self.dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            return RandomSampler(self.dataset_train)
