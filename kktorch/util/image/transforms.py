import random
from typing import Union
import numpy as np
import cv2
import torch
from PIL.Image import Image
from torchvision import transforms
from torchvision.transforms import functional

__all__ = [
    "ResizeFixRatio",
]


class _Transforms:
    def __init__(self, is_check_everytime: bool=True):
        self.is_check           = True
        self.is_check_everytime = is_check_everytime
    def __str__(self):
        return f'{self.__class__.__name__}({self.__dict__})'
    def __call__(self, img: Union[Image, np.ndarray, torch.Tensor], **kwargs):
        if self.is_check_everytime or self.is_check:
            self.check(img)
            self.is_check = False
        return self.transform(img, **kwargs)
    def check(self):
        # Only one call the first time.
        raise NotImplementedError
    def transform(self):
        raise NotImplementedError


class ResizeFixRatio(_Transforms):
    def __init__(self, size: int, fit_type: str="max", min_size: int=0, **kwargs):
        """
        Usage::
            >>> from PIL import Image
            >>> import numpy as np
            >>> img = Image.fromarray(np.random.randint(0, 255, (300, 500, 3)).astype(np.uint8))
            >>> ResizeFixRatio(100, "min")(img).size
            (166, 100)
            >>> ResizeFixRatio(100, "max")(img).size
            (100, 60)
            >>> ResizeFixRatio(200, "height")(img).size
            (333, 200)
            >>> ResizeFixRatio(200, "width")(img).size
            (200, 120)
            >>> ResizeFixRatio(200, "min", 400)(img).size
            (500, 300)
            >>> import torch
            >>> ResizeFixRatio(100, "min")(torch.rand(3, 200,200)).shape
            torch.Size([3, 100, 100])
        """
        assert isinstance(size,     int)
        assert isinstance(fit_type, str) and fit_type in ["max", "min", "height", "width"]
        assert isinstance(min_size, int) and min_size >= 0
        super().__init__(**kwargs)
        self.size      = size
        self.fit_type  = fit_type
        self.min_size  = min_size
        self.get_size  = lambda img: img.size
        self.resize    = lambda img, h, w: functional.resize(img, (h,w), interpolation=InterpolationMode.BILINEAR)
        self.calc_size = lambda w, h, size: (w, h)
        if   self.fit_type == "max":
            self.calc_size = lambda w, h, size: ((size, size * h / w, ) if w > h else (size * w / h, size, ))
        elif self.fit_type == "min":
            self.calc_size = lambda w, h, size: ((size * w / h, size) if w > h else (size, size * h / w, ))
        elif self.fit_type == "height":
            self.calc_size = lambda w, h, size: (size * w / h, size)
        elif self.fit_type == "width":
            self.calc_size = lambda w, h, size: (size, size * h / w)
    def check(self, img: Union[Image, np.ndarray, torch.Tensor]):
        assert isinstance(img, Image) or isinstance(img, np.ndarray) or isinstance(img, torch.Tensor)
        if   isinstance(img, np.ndarray):
            assert len(img.shape) == 3
            self.get_size = lambda img: (img.shape[1], img.shape[0])
            self.resize   = lambda img, h, w: cv2.resize(img, (w, h))
        elif isinstance(img, torch.Tensor):
            assert len(img.shape) == 3
            self.get_size = lambda img: (img.shape[2], img.shape[1])
    def transform(self, img: Union[Image, np.ndarray, torch.Tensor]) -> Union[Image, torch.Tensor]:
        w, h = self.get_size(img)
        if not (w < self.min_size or h < self.min_size):
            w, h = self.calc_size(w, h, self.size)
            w, h = int(w), int(h)
            return self.resize(img, h, w)
        return img