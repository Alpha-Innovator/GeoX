from PIL import Image
import torch
from torchvision import transforms

class ImageProcessor:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='train', image_size=224, interpolation=3, min_scale=0.8, max_scale=1.0):
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=interpolation,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=interpolation),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def preprocess(self, image, return_tensors='pt'):
        if isinstance(image, Image.Image):
            if return_tensors == 'pt':
                tensor = self.transform(image)
                return {'pixel_values': tensor.unsqueeze(0)}  # Add batch dimension
        raise TypeError("The input must be a PIL Image.")