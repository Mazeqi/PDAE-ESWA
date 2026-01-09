import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import cv2
import random

class MRART_Dataset(data.Dataset):


    training_file = 'training.pt'
    test_file = 'test.pt'


    def __init__(self, root, pollution, N, normal_class, task,seed
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = root
        self.normal_class = normal_class
        self.data=[]
        self.targets = []

        scores = pd.read_csv(root+ '/derivatives/scores.tsv', sep='\t')
        scores.columns=['id', 'targ']
        if task =='train':
            random.seed(seed)
            idx = np.where(np.array(scores.loc[scores['targ']==1]) == label_class)[0]
            random_ids = random.sample(list(range(0, len(idx))),N)
            for im in random_ids:
                self.data.append(root+ '/' + scores.iloc[im,0].split('_') [0] + '/anat/' + scores.iloc[im,0] + '.nii.gz')

                targ =torch.FloatTensor(torch.FloatTensor([0]))
                self.targets.append(targ)
        else:
            avoid = []
            names= []
            for im in self.indexes:
                names.append( scores.iloc[im,0].split('_') [0])

            for i,s in enumerate(scores['bids_name'].apply(lambda x: x.split('_') [0])):
                if s in names:
                    avoid.append(i)


            val_indexes =  [x for i,x in enumerate(scores.index.values) if (i not in self.indexes and i not in avoid) ]
            for im in val_indexes:
                self.data.append(root+ '/' + scores.iloc[im,0].split('_') [0] + '/anat/' + scores.iloc[im,0] + '.nii.gz')
                if scores.iloc[im,1] == 1:
                    targ = torch.FloatTensor([0])
                else:
                    targ =torch.FloatTensor([1])
                self.targets.append(targ)




    def __getitem__(self, index: int):


        index2 = int(np.floor(index/192))
        target = int(self.targets[index2])
        ind = index % 192
        img = self.data[index2]
        img = torch.FloatTensor(img)
        img = torch.stack((img,img,img),1)


    #    for im in img:
    #        transforms.Normalize([77.3,77.3,77.3], [140,140, 140], inplace=True)(im)
    #    img = transforms.Normalize([77.3,77.3,77.3], [140,140, 140])(img)

        img = img[ind,:,:,:]
        transforms.Normalize([77.3], [140], inplace=True)(img)
        return img, target, index2



    def __len__(self) -> int:
        if self.indexes == []:
            return len(self.data)
        else:
            return len(self.data) * 192
