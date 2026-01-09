import numpy as np
from skimage.metrics import structural_similarity
from skimage.util import img_as_ubyte
import cv2
import tqdm
import os
from matplotlib import pyplot as plt
import torch
import pandas as pd
from base.base_dataset import IMAGENET_MEAN, IMAGENET_STD
plt.switch_backend('agg')
import csv

def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """

    if len(imgs_input.shape) == 4:
        imgs_input_gray = []
        imgs_pred_gray = []

        if imgs_input.shape[-1] != 1:
            for i_image in imgs_input:
                i_image = cv2.cvtColor(i_image, cv2.COLOR_RGB2GRAY)
                imgs_input_gray.append(i_image)
                
            imgs_input_gray = np.array(imgs_input_gray).squeeze()
        else:
            imgs_input_gray = imgs_input.squeeze()
        if imgs_pred.shape[-1] != 1:
            for i_image in imgs_pred:
                i_image = cv2.cvtColor(i_image, cv2.COLOR_RGB2GRAY)
                imgs_pred_gray.append(i_image)
            imgs_pred_gray = np.array(imgs_pred_gray).squeeze()
        else:
            imgs_pred_gray = imgs_pred.squeeze()
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)
    return scores, resmaps

from skimage import filters
def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        #resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(1-score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred)**2
    scores = np.sqrt(np.sum(resmaps, axis=(1, 2)))
    return scores, resmaps

"""
    savefolder: the path to save images
    ori_image: 
            list with normalize images or numpy with normalize images,
            shape like [samples, h, w, c]
    rec_imgs: 
            list with normalize images or numpy with normalize images,
            shape like [samples, h, w, c]
    mask_gts: 
            mask of the groud true with normilization
            shape like [samples, h, w]

    mask_pred: 
            mask_pred obtained from the model,
            shape like [samples, h, w]
"""  
def plot_segmentation_images(
    savefolder,
    ori_images,
    rec_imgs,
    mask_gts,
    mask_pred,
    denorm = True
):

    os.makedirs(savefolder, exist_ok=True)
    if denorm == False:
        in_std = 1
        in_mean = 0
    else:
        in_std = np.array(IMAGENET_STD).reshape(1, 1, 3)
        in_mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)

    ori_images = np.clip(
        (np.array(ori_images) * in_std + in_mean) * 255, 0, 255
    ).astype(np.uint8)

    
    rec_imgs = np.clip(
        (np.array(rec_imgs) * in_std + in_mean) * 255, 0, 255
    ).astype(np.uint8)

    mask_pred = np.clip(
        (np.array(mask_pred) * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255, 0, 255
    ).astype(np.uint8)

    #threshold = (threshold*IMAGENET_STD[0]+IMAGENET_MEAN[0])*255
    #mask_pred[mask_pred <= threshold] = 0

    mask_gts = np.array(mask_gts)
    ind = 0
    for image, rec_img, mask, segmentation in tqdm.tqdm(
        zip(ori_images, rec_imgs, mask_gts, mask_pred),
        desc="Plot Images...",
        leave=True,
        position=1
    ):
        savename = str(ind) + '.jpg'
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 4)
        axes[0].imshow(image)
        axes[1].imshow(rec_img)
        axes[2].imshow(mask)
        axes[3].imshow(segmentation)
        f.tight_layout()
        f.savefig(savename)
        plt.close()
        ind = ind + 1

"""
    savefolder: the path to save images
    ori_image: 
            list with normalize images or numpy with normalize images,
            shape like [samples, h, w, c]
    rec_imgs: 
            list with normalize images or numpy with normalize images,
            shape like [samples, h, w, c]
"""   
def plot_reconstruct_images(
    savefolder,
    ori_images,
    rec_imgs,
    denorm = True
):

    os.makedirs(savefolder, exist_ok=True)
    in_std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    in_mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    if denorm == False:
        in_std = 1
        in_mean = 0

    ori_images = np.clip(
        (np.array(ori_images) * in_std + in_mean) * 255, 0, 255
    ).astype(np.uint8)

    
    rec_imgs = np.clip(
        (np.array(rec_imgs) * in_std + in_mean) * 255, 0, 255
    ).astype(np.uint8)

    ind = 0
    for image, rec_img in zip(ori_images, rec_imgs):
        
        savename = str(ind) + '.jpg'
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[1].imshow(rec_img)
        f.tight_layout()
        f.savefig(savename)
        plt.close()
        ind = ind + 1

def convert2img(image, imtype=np.uint8):
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
        assert len(image.squeeze().shape) < 4
    if image.dtype != np.uint8:
        image = (image.squeeze() * 0.5 + 0.5) * 255
    return image.astype(imtype)


"""
    savefolder: the path to save images
    images_dict: 
            dict with normalize images list of numpy,
            shape like [samples, h, w, c]

"""  
def plot_images(
    savefolder,
    images_dict:dict,
    denorm = True
):

    os.makedirs(savefolder, exist_ok=True)
    if denorm == False:
        in_std = 1
        in_mean = 0
    else:
        in_std = np.array(IMAGENET_STD).reshape(1, 1, 3)
        in_mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)

    num_list = len(images_dict.keys())
    name_list = list(images_dict.keys())
    ran_key = name_list[0]
    num_image = len(images_dict[ran_key])
    
    image_num_list = range(0, num_image)
    with tqdm.tqdm(image_num_list, desc="save images...", position=1, leave=False
                           ) as image_num_list_loader:
        for i_img_ind in image_num_list_loader:
            save_ind = 0
            f, axes = plt.subplots(1, num_list)
            for i_key in images_dict.keys():
                image_list = images_dict[i_key]
                i_image = image_list[i_img_ind]

                if denorm == True:
                    if len(i_image.shape) == 3 and i_image.shape[-1] == 3:
                        i_image = np.clip(
                            (np.array(i_image) * in_std + in_mean) * 255, 0, 255
                        ).astype(np.uint8)
                    #else:
                    #    i_image = np.clip(
                    #        (np.array(i_image) * IMAGENET_STD[0] + IMAGENET_MEAN[0]) * 255, 0, 255
                    #    ).astype(np.uint8)


                axes[save_ind].imshow(i_image)
                axes[save_ind].set_title(i_key, fontsize=8)
                save_ind = save_ind + 1

            savename = str(i_img_ind) + '.jpg'
            savename = os.path.join(savefolder, savename)
            f.tight_layout()
            f.savefig(savename)
            plt.close()


def write_results(results:dict, cur_class, total_classes, csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])
            df_temp.index = df_temp.index.astype(str)
        
            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')
        
    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')

if __name__ == "__main__":
    results = dict()
    results[f'i_roc'] = 1
    results[f'p_roc'] = 1
    results[f'p_pro'] = 1
    total_classes = ["bottle", "cable"]
    write_results(
            results,
            "bottle",
            total_classes,
            "output/test/mvtec/result.csv"
    )
