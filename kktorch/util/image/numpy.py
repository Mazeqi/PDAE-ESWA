import cv2
from kktorch.util.com import check_type_list
import numpy as np
from PIL import Image
from typing import List


__all__ = [
    "fit_resize",
    "pil2cv",
    "cv2pil",
    "gray_to_rgb",
    "concat_images",
]


def fit_resize(img: np.ndarray, dim: str, scale: int):
    """
    Params::
        img: image
        dim: x or y
        scale: width or height
    """
    if dim not in ["x","y"]: raise Exception(f"dim: {dim} is 'x' or 'y'.")
    height = img.shape[0]
    width  = img.shape[1]
    height_after, width_after = None, None
    if   type(scale) == int and scale > 10:
        if   dim == "x":
            width_after  = int(scale)
            height_after = int(height * (scale / width))
        elif dim == "y":
            height_after = int(scale)
            width_after  = int(width * (scale / height))
    else:
        raise Exception(f"scale > 10.")
    img = cv2.resize(img , (width_after, height_after)) # w, h
    return img

def pil2cv(img: Image) -> np.ndarray:
    new_image = np.array(img, dtype=np.uint8)
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(img: np.ndarray):
    new_image = img.copy()
    if new_image.ndim == 2:  # gray
        pass
    elif new_image.shape[2] == 3:  # RGB
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # RGBA
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def gray_to_rgb(img: np.ndarray):
    return np.concatenate([img.reshape(img.shape[0], img.shape[1], 1).copy() for _ in range(3)], axis=2)

def concat_images(imgs: List[np.ndarray], shape: (int, int), resize: (int, int)=None):
    """
    Params::
        imgs: List of imgs
        shape: height, width
        resize: width, height
    """
    assert check_type_list(imgs, np.ndarray)
    assert isinstance(shape, list) or isinstance(shape, tuple)
    shape = list(shape)
    assert len(shape) == 2 and check_type_list(shape, int)
    assert len(imgs) >= (shape[0] * shape[1])
    if resize is not None:
        assert isinstance(resize, list) or isinstance(resize, tuple)
        resize = list(resize)
        assert check_type_list(resize, int)
    img = None
    for i in range(shape[0]):
        imgwk = None
        for j in range(shape[1]):
            imgwkwk = imgs[i * shape[1] + j]
            if resize is not None: imgwkwk = cv2.resize(imgwkwk, resize)
            if imgwk is None: imgwk = imgwkwk
            else: imgwk = np.concatenate([imgwk, imgwkwk], axis=1)
        if img is None: img = imgwk
        else: img = np.concatenate([img, imgwk], axis=0)
    return img
