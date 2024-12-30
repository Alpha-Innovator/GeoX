import os
import copy
import json
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers



class GeoQADataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        """
        Args:
            data_path (str): The path to the dataset.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            data_args (DataArguments): Additional data arguments for preprocessing.
        """
        super(GeoXDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = json.load(open(self.data_args.data_path, "r"))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """Returns the lengths of the samples."""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """Returns the modality-specific lengths of the samples."""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def preprocess_multimodal(self, sources):
        """
        Preprocesses the multimodal data.
        Args:
            sources (list): A list of source data.
        Returns:
            dict: Preprocessed multimodal data.
        """
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)

        return sources

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Fetches the item from the dataset at index i.
        Args:
            i (int): Index of the sample.
        Returns:
            Dict[str, torch.Tensor]: Tokenized data with optional image data.
        """
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

            # Preprocess image
            if self.data_args.image_aspect_ratio == 'pad':
                assert 0  # padding logic can be added if needed
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = self.preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]))
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # Preprocess and tokenize
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Include image data if available
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            assert 0  # Multimodal model but no image, you can raise an error or handle this case

        return data_dict
