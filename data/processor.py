import re
from PIL import Image
from torchvision import transforms

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()



class ImageProcessor:
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        mode='train',
        image_size=224,
        interpolation=3,
        min_scale=0.8,
        max_scale=1.0
    ):
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

    def __call__(self, item):
        return self.transform(item)

    def preprocess(self, image, return_tensors='pt'):
        if isinstance(image, Image.Image):
            if return_tensors == 'pt':
                tensor = self.__call__(image)  # 调用 __call__ 方法
                return {'pixel_values': tensor.unsqueeze(0)}  # 添加批次维度
        raise TypeError("The input must be a PIL Image.")


    
class TextProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=64):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        if type(caption) is list:
            caption = [self.prompt + self.pre_caption(caption[0]), self.pre_caption(caption[1])]
        else:
            caption = self.prompt + self.pre_caption(caption)

        return caption


    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

