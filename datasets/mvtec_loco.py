import os
import sys
sys.path.append("./")
sys.path.append("../")

import PIL
import torch
from torchvision import transforms
from base.base_dataset import DatasetSplit, IMAGENET_MEAN, IMAGENET_STD
import numpy as np
from datasets.CDO_Gen_Ano import CDO_GEN_ANO
from datasets.cutpaste import CutPaste
from datasets.cutpaste_ori import CutPasteOri
from .src.DefectMaker import defect_makers as dm
from .src.DefectMaker.base import *
from .rand_mask import rand_mask
_CLASSNAMES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]

class MvtecLocoDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Mvtec_loco.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        dataset_para = {},
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.dataset_para = dataset_para
        self.normal_label = "good"
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        

        if self.split == DatasetSplit.TRAIN:
            if "cutpaste" in dataset_para and dataset_para["cutpaste"]["mode"] == True:
                self.cutpast_gen_ano = CutPaste(cutpaste_area_ratio_list=dataset_para["cutpaste"]["cutpaste_noise"]["mvtec_loco"])
            elif "cutpaste_ori" in dataset_para and dataset_para["cutpaste_ori"]["mode"] == True:
                self.cutpast_gen_ano = CutPasteOri(type=dataset_para["cutpaste_ori"]["type"])
            elif "defect_maker" in dataset_para and dataset_para["defect_maker"]["mode"] == True: 
                make_parm = {"saliency_method":SalienMethod.IMAGE_FIT, "fuse_weight_range" :[0.5, 0.9]}
                defect_maker = dm.get_defect_maker(dataset="", fill_param={}, shape_param= {}, fill_name="RandomNoiseFillMaker", shape_name="BezierShapeMaker", make_param=make_parm)
                self.cutpast_gen_ano = defect_maker._make_defct_img
            elif "rand_mask" in dataset_para and dataset_para["rand_mask"]["mode"] == True: 
                self.cutpast_gen_ano = rand_mask
            
            self.transform_img  = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform_img = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path_list = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        augmented_image = None
        anomaly_mask = None

        if self.split == DatasetSplit.TRAIN :
            if "aug_mode" in self.dataset_para and self.dataset_para["aug_mode"] == True:
                if "rand_mask" in self.dataset_para and self.dataset_para["rand_mask"]["mode"] == True: 
                    
                    augmented_image = self.transform_img(image)
                    augmented_image, anomaly_mask = self.cutpast_gen_ano(augmented_image, self.dataset_para["rand_mask"]["mask_ratio"])
                else:
                    augmented_image, anomaly_mask = self.cutpast_gen_ano(image)
                    augmented_image = self.transform_img(augmented_image)
                    anomaly_mask = self.transform_mask(anomaly_mask)

        else:
            augmented_image = PIL.Image.open(image_path).convert("RGB")
            augmented_image = self.transform_img(augmented_image)
            anomaly_mask = torch.zeros([1, *augmented_image.size()[1:]])

        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path_list is not None:
            #mask = PIL.Image.open(mask_path_list[0])
            #mask = self.transform_mask(mask)

            #mask = PIL.Image.open(mask_path_list[0])
            #mask = self.transform_mask(mask)
            mask = 0
            for i_mask_path in mask_path_list:
                i_mask = PIL.Image.open(i_mask_path)
                
                i_mask = self.transform_mask(i_mask)
                mask += i_mask
            #str_name = str(random.random())
            #mask.save("output/test/im/" + str_name +".png")
            mask[mask>0]=1
        else:
            mask = torch.zeros([1, *image.size()[1:]])
            
        if self.split == DatasetSplit.TRAIN:   

            if augmented_image != None: 
                redict = {
                    "image": image,
                    "mask": mask,
                    "aug_image":augmented_image,
                    "anomaly_mask": anomaly_mask,
                    "classname": classname,
                    "anomaly": anomaly,
                    "label": int(anomaly != self.normal_label),
                    "image_name": "/".join(image_path.split("/")[-4:]),
                    "image_path": image_path,
                }
            else:
                redict = {
                    "image": image,
                    "mask": mask,
                    "classname": classname,
                    "anomaly": anomaly,
                    "label": int(anomaly != self.normal_label),
                    "image_name": "/".join(image_path.split("/")[-4:]),
                    "image_path": image_path,
                }
        else:
            redict = {
                "image": image,
                "mask": mask,
                "classname": classname,
                "anomaly": anomaly,
                "label": int(anomaly != self.normal_label),
                "image_name": "/".join(image_path.split("/")[-4:]),
                "image_path": image_path,
            }
        return redict

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)
            #anomaly_types =  ["good", "logical_anomalies"]
            #anomaly_types =  ["good", "structural_anomalies"]
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            for anomaly in anomaly_types:
                #if anomaly == "good":
                #    continue
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != self.normal_label:
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_dirs = sorted(os.listdir(anomaly_mask_path))
                    #maskpaths_per_class[classname][anomaly] = [
                    #    os.path.join(anomaly_mask_path, x, "000.png") for x in anomaly_mask_files
                    #]
                    maskpaths_per_class[classname][anomaly] = {}
                    for i_dir in anomaly_mask_dirs:
                        anomaly_mask_files = sorted(os.listdir(os.path.join(anomaly_mask_path, i_dir)))
                        maskpaths_per_class[classname][anomaly][i_dir] = [
                            os.path.join(anomaly_mask_path, i_dir, x) for x in anomaly_mask_files
                        ]
                    
                else:
                    maskpaths_per_class[classname][self.normal_label] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != self.normal_label:
                        mask_dir_list = sorted(maskpaths_per_class[classname][anomaly].keys())
                        data_tuple.append(maskpaths_per_class[classname][anomaly][mask_dir_list[i]])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
    
    