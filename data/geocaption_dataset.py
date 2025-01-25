import os
import json
import random
import time
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from typing import Iterable

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]):
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            if any(ext in ann_path for ext in ['csv', 'tsv']):
                df = pd.read_csv(ann_path)
                self.annotation.extend(df.to_dict(orient="records"))
            elif 'jsonl' in ann_path:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f])
            else:
                with open(ann_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                        self.annotation.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v} for k, v in loaded.items()])

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys()
        for k in keys:
            values = [sample[k] for sample in samples]
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        all_keys = set()
        for s in samples:
            all_keys.update(s)
        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())
        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})
        return self.datasets[0].collater(samples_shared_keys)

class GeoCapDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        return {
            "image": image,
            "text_input": caption,
            "image_id": ann["image_id"]
        }

