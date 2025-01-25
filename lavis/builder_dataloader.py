"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import random
import time
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Iterable
import logging

import shutil
import warnings
from data.geocaption_dataset import (
    GeoCapDataset,
)
from lavis.common.registry import registry


import lavis.common.utils as utils
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized
from data.processor import BaseProcessor, ImageProcessor, TextProcessor
from omegaconf import OmegaConf

class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}



    def build_datasets(self):

        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        self.vis_processors["train"] = ImageProcessor(mode='train')
        self.vis_processors["eval"] = ImageProcessor(mode='val')
        self.text_processors["train"] = TextProcessor()
        

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])



    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)


        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            
            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets


class MultiModalDatasetBuilder(BaseDatasetBuilder):
    """
    MultiModalDatasetBuilder is a utility class designed to construct datasets
    suitable for multi-modal tasks. This class simplifies the creation of 
    datasets that incorporate data of multiple modalities, such as text, 
    images, video, or audio.
    """
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if isinstance(self.data_type, str):
            self.data_type = [self.data_type]


    def build_processors(self):
        self.vis_processors["train"] = ImageProcessor(mode='train')
        self.vis_processors["eval"] = ImageProcessor(mode='val')
        self.text_processors["train"] = TextProcessor()
        
    def _get_absolute_path(self, path):
        if not os.path.isabs(path):
            return utils.get_cache_path(path)
        return path

    def build(self):
        self.build_processors()
        build_info = self.config.build_info
        datasets = {}
        
        for split, info in build_info.annotations.items():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"
            dataset_args = self._get_dataset_args(info, is_train)
            
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(**dataset_args)

        return datasets

    def _get_dataset_args(self, info, is_train):
        dataset_args = dict(self.config.build_info.get('kwargs', {}))
        
        for modality in self.data_type:
            proc_name = f"{'vis' if 'image' in modality else modality}_processor"
            dataset_args[proc_name] = self.processors["train" if is_train else "eval"][modality]
            mm_path = self._get_absolute_path(self.config.build_info.get(modality).storage)
            dataset_args[f"{'vis' if 'image' in modality  else modality}_root"] = mm_path
        
        dataset_args['text_processor'] = self.text_processors["train" if is_train else "eval"]
        dataset_args["ann_paths"] = [self._get_absolute_path(path) for path in info.storage]
        dataset_args['modalities'] = self.data_type
        
        # Conform to base
        for key in ['vis_processor', 'vis_root', 'test_processor']:
            dataset_args.setdefault(key, None)
        
        return dataset_args

def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    return next(iter(cfg.values()))

class MultiIterLoader:
    def __init__(self, loaders, ratios=None):
        for loader in loaders:
            assert hasattr(loader, "__next__"), "Loader {} has no __next__ method.".format(loader)
        if ratios is None:
            ratios = [1.0] * len(loaders)
        else:
            assert len(ratios) == len(loaders)
            ratios = [float(ratio) / sum(ratios) for ratio in ratios]
        self.loaders = loaders
        self.ratios = ratios

    def __next__(self):
        loader_idx = random.choices(range(len(self.loaders)), self.ratios, k=1)[0]
        return next(self.loaders[loader_idx])

class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch
            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None and batch is not {}:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method

def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, (list, tuple)):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)

class IterLoader:
    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, (list, tuple)):
        return [move_to_cuda(t) for t in batch]
    elif isinstance(batch, dict):
        return {k: move_to_cuda(v) for k, v in batch.items()}
    else:
        return batch

@registry.register_builder("geo_caption")
class GeoCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = GeoCapDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/data.yaml",
    }


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("geo_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
