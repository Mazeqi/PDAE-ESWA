import pandas as pd
import cv2
from typing import List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

from kktorch.util.com import check_type_list
import kktorch.util.image as tfms


__all__ = [
    "ImageDataset",
    "DataframeDataset",
]


class ImageDataset(Dataset):
    def __init__(
        self, image_paths: List[str], labels: List[object],
        transforms: Union[tfms.Compose, List[tfms.Compose]]=tfms.Compose([tfms.ToTensor(),])
    ):
        super().__init__()
        """
        Params::
            image_paths:
                list of image file paths.
                ["./data/imageA.jpg", "./data/imageB.jpg", ...]
            labels:
                labels.
                [1, 0, ...] or [[1,2,3], [2,3,4], ...] or [objectA, objectB, ...]
            transforms:
                torchvision.transforms.Compose or List
                If Compose:
                    a image is transformed by Compose and create 1 image.
                If List:
                    a image is transformed by Compose of List and create N image.
        """
        assert check_type_list(image_paths, str)
        if labels is None: labels = [0] * len(image_paths)
        assert isinstance(labels, list) and len(image_paths) == len(labels)
        assert isinstance(transforms, tfms.Compose) or check_type_list(transforms, tfms.Compose)
        self.image_paths  = image_paths
        self.labels       = labels
        self.len          = len(self.image_paths)
        self.transforms   = [transforms] if isinstance(transforms, tfms.Compose) else transforms
        self.concat       = (lambda x: x) if len(self.transforms) == 1 else self.concat_same_resolution
        self.is_check     = True
        self.dict_indexes = {}
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        path = self.image_paths[index]
        img  = Image.open(path)
        imgs = []
        for procs in self.transforms: imgs.append(procs(img))
        imgs   = self.concat(imgs)
        labels = self.labels[index]
        return (*imgs, labels)
    def concat_same_resolution(self, input: List[torch.Tensor]):
        """
        Combine images of the same resolution when there are multiple augmentations and different resolutions are output.
        """
        if self.is_check:
            assert check_type_list(input, torch.Tensor)
            assert sum([(len(img.shape) == 3) for img in input]) == len(input)
            for i, img in enumerate(input):
                shape = tuple(img.shape)
                if self.dict_indexes.get(shape) is None:
                    self.dict_indexes[shape] = [i]
                else:
                    self.dict_indexes[shape].append(i)
            self.is_check = False
        output = []
        for _, y in self.dict_indexes.items():
            output.append(torch.stack([input[i] for i in y]))
        return output
    def get_image(self, index: int, img_type: str="cv2"):
        assert isinstance(img_type, str) and img_type in ["cv2", "pil"]
        path = self.image_paths[index]
        img  = None
        if   img_type == "cv2":
            img = cv2.imread(path)
        elif img_type == "pil":
            img = Image.open(path)
        return img
    def calc_mean_and_std(self):
        img = self.get_image(0, img_type="cv2").astype(float) / len(self)
        H, W, _ = img.shape
        for i in range(1, len(self)):
            imgwk = self.get_image(i, img_type="cv2")
            if (H, W) != imgwk.shape[:2]:
                imgwk = cv2.resize(imgwk, (W, H))
            img += imgwk.astype(float) / len(self)
        img = img.transpose(2,0,1).reshape(3, -1) / 255.
        return img.mean(-1).tolist(), img.std(-1).tolist()


class DataframeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, columns: Union[List[str], List[int]], **kwargs):
        assert isinstance(df, pd.DataFrame)
        assert check_type_list(columns, [int, str])
        self.columns = columns
        self.ndf     = df.iloc[:, self.columns].values.copy() if check_type_list(columns, [int]) else df.loc[:, self.columns].values.copy()
        super().__init__(**kwargs)
    def __getitem__(self, index: int):
        return tuple(self.ndf[index])
    def __len__(self):
        return self.ndf.shape[0]
