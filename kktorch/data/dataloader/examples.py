import random, os
import zipfile, tarfile
from typing import Callable, Union, List
import pandas as pd
import numpy as np
import torch, torchvision

import kktorch
from kktorch.data.dataset import DataframeDataset, ImageDataset
from kktorch.util.com import check_type_list, correct_dirpath, makedirs
from kktorch.util.files import download_file
from kktorch.util.dataframe import text_files_to_dataframe
from kktorch.data.dataloader.dataloader import BaseDataLoader, TextDataLoader
import kktorch.util.image as tfms
from kktorch.util.com import get_file_list


__all__ = [
    "MNISTDataLoader",
    "PASCALvoc2012DataLoader",
    "NewsPaperDataLoader",
    "LivedoorNewsDataLoader",
    "MVTecADDataLoader",
]


ROOTDATADIR=f"{correct_dirpath(kktorch.__path__[0])}__data__"


def split_train_test(index: int, seed: int=0, percent: float=0.8):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    indexes = np.random.permutation(np.arange(index))
    size    = int(indexes.shape[0] * percent)
    indexes_train = indexes[:size ]
    indexes_test  = indexes[ size:]
    return indexes_train, indexes_test


def deploy_files(url: str, dirpath: str, download: bool, extract: str="zip"):
    assert isinstance(extract, str) and extract in ["zip", "gz", "tar"]
    makedirs(dirpath, exist_ok=True, remake=False)
    filepath = dirpath + os.path.basename(url)
    if download and not os.path.exists(filepath):
        filepath = download_file(url, filepath=filepath)
        if extract == "zip":
            with zipfile.ZipFile(filepath) as existing_zip:
                existing_zip.extractall(dirpath)
        elif extract == "gz":
            with tarfile.open(filepath, 'r:gz') as t:
                t.extractall(path=dirpath)
        elif extract == "tar":
            with tarfile.open(filepath, 'r') as t:
                t.extractall(path=dirpath)


class MNISTDataLoader(BaseDataLoader):
    """
    Usage::
        >>> from kktorch.data.dataloader import MNISTDataLoader
        >>> dataloader_train = MNISTDataLoader(root='./data', train=True,  download=True, batch_size=2, shuffle=True)
        >>> batch, label = next(iter(dataloader_train))
        >>> batch
        tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                ...,
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.]]],


                [[[0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                ...,
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.],
                [0., 0., 0.,  ..., 0., 0., 0.]]]])
        >>> batch.shape
        torch.Size([2, 1, 28, 28])
        >>> label
        tensor([2, 5])
        >>> label.shape
        torch.Size([2])
    """
    def __init__(
        self, root: str=ROOTDATADIR, train: bool=True, download: bool=True, 
        transforms=[tfms.ToTensor(), ], 
        dtype_data=torch.float32, dtype_target=torch.long, 
        classes_targets: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        **kwargs
    ):
        assert isinstance(classes_targets, list) and check_type_list(classes_targets, int)
        transforms = transforms if isinstance(transforms, list) else []
        transforms = tfms.Compose(transforms)
        dataset    = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transforms)
        targets    = dataset.targets.numpy().copy()
        targets    = torch.from_numpy(np.isin(targets, classes_targets))
        dataset.data    = dataset.data[   targets, :, :]
        dataset.targets = dataset.targets[targets]
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)


class PASCALvoc2012DataLoader(BaseDataLoader):
    PASCALVOC2012_DEFAULT_MEAN = (0.4587, 0.4380, 0.4017)
    PASCALVOC2012_DEFAULT_STD  = (0.2749, 0.2722, 0.2870)
    def __init__(
        self, root: str=ROOTDATADIR, train: bool=True, download: bool=True, 
        transforms: Union[tfms.Compose, List[tfms.Compose]]=tfms.Compose([
            tfms.ToTensor(),
        ]), is_label_binary_class: bool=False,
        dtype_data=torch.float32, dtype_target=torch.long,
        **kwargs
    ):
        """
        see: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
        Labels::
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html
            There are three ground truth labels:
            -1: Negative: The image contains no objects of the class of interest. A classifier should give a `negative' output.
             1: Positive: The image contains at least one object of the class of interest. A classifier should give a `positive' output.
             0: ``Difficult'': The image contains only objects of the class of interest marked as `difficult'.

        Usage Simple::
            >>> from kktorch.data.dataloader import PASCALvoc2012DataLoader
            >>> dataloader_train = PASCALvoc2012DataLoader(
                    root='./data', train=True, download=True, batch_size=2, shuffle=False, drop_last=False, num_workers=1,
                )
            >>> dataloader_train[0]
            (tensor([[[ [0.2275, 0.2078, 0.2157,  ..., 0.2431, 0.2745, 0.2471],
                        [0.2588, 0.2078, 0.1961,  ..., 0.2706, 0.2314, 0.2275],
                        [0.2588, 0.1725, 0.2039,  ..., 0.2667, 0.2549, 0.2510],
                        ...,
                        [0.3333, 0.3569, 0.3294,  ..., 0.2118, 0.2706, 0.2549],
                        [0.3686, 0.3216, 0.3255,  ..., 0.2353, 0.2510, 0.2275],
                        [0.3765, 0.3216, 0.2941,  ..., 0.2275, 0.1765, 0.1804]],

                        [[0.2275, 0.2157, 0.2353,  ..., 0.3529, 0.3804, 0.3529],
                        [0.2392, 0.2078, 0.2078,  ..., 0.3765, 0.3608, 0.3569],
                        [0.2353, 0.1569, 0.2078,  ..., 0.3725, 0.3686, 0.3647],
                        ...,
                        [0.2549, 0.2902, 0.2745,  ..., 0.2118, 0.2196, 0.2039],
                        [0.2863, 0.2549, 0.2706,  ..., 0.2431, 0.2353, 0.2118],
                        [0.2941, 0.2549, 0.2392,  ..., 0.2353, 0.1922, 0.1961]],

                        [[0.2275, 0.2039, 0.2118,  ..., 0.3647, 0.4157, 0.3882],
                        [0.2549, 0.2078, 0.1804,  ..., 0.4039, 0.3882, 0.3843],
                        [0.2353, 0.1529, 0.1882,  ..., 0.4078, 0.4000, 0.3961],
                        ...,
                        [0.2275, 0.2588, 0.2392,  ..., 0.1725, 0.1882, 0.1725],
                        [0.2667, 0.2275, 0.2353,  ..., 0.2000, 0.1922, 0.1686],
                        [0.2824, 0.2275, 0.1961,  ..., 0.1922, 0.1373, 0.1412]]]]), 
                        tensor([[-1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]))
        Usage with Augmentation::
            >>> from torchvision import transforms
            >>> from PIL import Image 
            >>> from kktorch.data.dataloader import PASCALvoc2012DataLoader
            >>> dataloader_train = PASCALvoc2012DataLoader(
                    root='./data', train=True, download=True, batch_size=2, shuffle=False, drop_last=False, num_workers=1,
                    transforms=transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_MEAN, 
                            PASCALvoc2012DataLoader.PASCALVOC2012_DEFAULT_STD
                        ),
                    ])
                )
        """
        self.dirpath = correct_dirpath(root) + "PASCALvoc2012/"
        # download
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
        deploy_files(url, self.dirpath, download, extract="tar")
        # set data infomation
        df = pd.DataFrame(get_file_list(self.dirpath + "VOCdevkit/VOC2012/JPEGImages/"), columns=["filepath"])
        df["filename"] = df["filepath"].apply(lambda x: os.path.basename(x))
        df["id"]       = df["filename"].str.replace(".jpg", "", regex=False)
        for x in get_file_list(self.dirpath + "VOCdevkit/VOC2012/ImageSets/Main", regex_list=[r"_trainval\.txt"]):
            dfwk = pd.read_csv(x, sep="\s+", header=None, names=["id", "label_" + os.path.basename(x).split("_")[0]])
            df   = pd.merge(df, dfwk, how="left", on="id")
        # train, test
        indexes_train = pd.read_csv(self.dirpath + "VOCdevkit/VOC2012/ImageSets/Main/train.txt", header=None)[0].values
        indexes_test  = pd.read_csv(self.dirpath + "VOCdevkit/VOC2012/ImageSets/Main/val.txt",   header=None)[0].values
        df_train = df.loc[df["id"].isin(indexes_train)].copy()
        df_test  = df.loc[df["id"].isin(indexes_test) ].copy()
        self.df  = df_train if train else df_test
        if is_label_binary_class:
            columns = self.df.columns[df.columns.str.contains("^label_")]
            self.df.loc[:, columns] = self.df.loc[:, columns].replace(0, 1).replace(-1, 0).astype(int)
        dataset  = ImageDataset(
            self.df["filepath"].values.tolist(), 
            self.df.loc[:, self.df.columns[self.df.columns.str.contains("^label_")].tolist()].astype(int).values.tolist(),
            transforms=transforms
        )
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)


class NewsPaperDataLoader(TextDataLoader):
    def __init__(
        self, tokenizer: Callable, root: str=ROOTDATADIR, train: bool=True, download: bool=True, **kwargs
    ):
        """
        see: https://archive.ics.uci.edu/ml/datasets/News+Aggregator
        CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)
        replace category name to {"b": 0, "t": 1, "e": 2, "m": 3}
        ex)
            text : 'Fed official says weak data caused by weather, should not slow taper'
            label: 'b'
        """
        self.dirpath = correct_dirpath(root) + "NewsPaper/"
        # download
        url          = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
        csv_filepath = self.dirpath + "newsCorpora.csv"
        deploy_files(url, self.dirpath, download, extract="zip")
        # set data infomation
        df    = pd.read_csv(csv_filepath, sep="\t", header=None)
        df[4] = df[4].map({"b": 0, "t": 1, "e": 2, "m": 3}).astype(np.int32)
        # train, test
        indexes_train, indexes_test = split_train_test(df.shape[0])
        df_train = df.iloc[indexes_train].copy()
        df_test  = df.iloc[indexes_test ].copy()
        dataset  = DataframeDataset(df_train if train else df_test, columns=[1, 4])
        super().__init__(dataset, dtype_data=torch.long, dtype_target=torch.long, tokenizer=tokenizer, **kwargs)


class LivedoorNewsDataLoader(TextDataLoader):
    def __init__(
        self, tokenizer: Callable, root: str=ROOTDATADIR, train: bool=True, download: bool=True, columns=["text", "label"], **kwargs
    ):
        """
        see: https://www.rondhuit.com/download.html#ldcc
        label: {
            'dokujo-tsushin': 0, 'it-life-hack': 1, 'kaden-channel': 2,
            'livedoor-homme': 3, 'movie-enter':4, 'peachy':5, 'smax': 6,
            'sports-watch': 7, 'topic-news': 8
        }
        ex)
            text:  '吉田麻也、サウサンプトンFC移籍が決定\nサッカー日本代表・吉田麻也のサウサンプトンFC移籍が正式に決定した。\n\n同クラブは、公式サイトで「セインツ（サウサンプトンFCの愛称）にとって喜ばしい発表」として、吉田の入団を発表。オフィシャルツイッター（@officialsaints）上でも、「Japanese international Maya Yoshida is set to join #SaintsFC on a three-year deal from VVV-Venlo.」と呟いている。\n\nまた、同ツイッターでは、「プレミアリーグにくることは、子供の頃からの大きな夢の一つ」という吉田のコメントも掲載。オランダ・VVVフェンローからの移籍となった吉田の契約期間は3年となる。\n\n一部には、吉田のサウサンプトンFC入団が決まれば、9月2日のマンU戦でのプレミアデビュー＆香川真司との日本人対決実現という報道もあったが、その可能性も限りなく高まったといえるだろう。日本のファンからすれば、また大きな楽しみが増える吉田のプレミア入りだ。\n'
            site:  sports-watch
            title: '吉田麻也、サウサンプトンFC移籍が決定'
            body:  'サッカー日本代表・吉田麻也のサウサンプトンFC移籍が正式に決定した。\n\n同クラブは、公式サイトで「セインツ（サウサンプトンFCの愛称）にとって喜ばしい発表」として、吉田の入団を発表。オフィシャルツイッター（@officialsaints）上でも、「Japanese international Maya Yoshida is set to join #SaintsFC on a three-year deal from VVV-Venlo.」と呟いている。\n\nまた、同ツイッターでは、「プレミアリーグにくることは、子供の頃からの大きな夢の一つ」という吉田のコメントも掲載。オランダ・VVVフェンローからの移籍となった吉田の契約期間は3年となる。\n\n一部には、吉田のサウサンプトンFC入団が決まれば、9月2日のマンU戦でのプレミアデビュー＆香川真司との日本人対決実現という報道もあったが、その可能性も限りなく高まったといえるだろう。日本のファンからすれば、また大きな楽しみが増える吉田のプレミア入りだ。\n'
        """
        self.dirpath = correct_dirpath(root) + "LivedoorNews/"
        url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
        deploy_files(url, self.dirpath, download, extract="gz")
        filepath_df = self.dirpath + "df.pickle"
        if os.path.exists(filepath_df):
            df = pd.read_pickle(filepath_df)
        else:
            df = text_files_to_dataframe(self.dirpath + "text/", regex_list=[r"[0-9]\.txt$"], encoding="utf8")
            df.to_pickle(filepath_df)
        df["text"]  = df["text"].apply(lambda x: "\n".join(x.split("\n")[2:]))
        df["title"] = df["text"].apply(lambda x: x.split("\n")[0])
        df["body"]  = df["text"].apply(lambda x: "\n".join(x.split("\n")[1:]))
        df["site"]  = df["filename"].apply(lambda x: "-".join(x.split("-")[:-1]))
        df["label"] = df["site"].map({
            'dokujo-tsushin': 0, 'it-life-hack': 1, 'kaden-channel': 2,
            'livedoor-homme': 3, 'movie-enter':4, 'peachy':5, 'smax': 6,
            'sports-watch': 7, 'topic-news': 8
        })
        # train, test
        indexes_train, indexes_test = split_train_test(df.shape[0])
        df_train = df.iloc[indexes_train].copy()
        df_test  = df.iloc[indexes_test ].copy()
        self.df  = df_train if train else df_test
        dataset  = DataframeDataset(df_train if train else df_test, columns=columns)
        super().__init__(dataset, dtype_data=torch.long, dtype_target=torch.long, tokenizer=tokenizer, **kwargs)


class MVTecADDataLoader(BaseDataLoader):
    MVTecAD_DEFAULT_MEAN = {
        "capsule": (0.6540, 0.6668, 0.6968)
    }
    MVTecAD_DEFAULT_STD  = {
        "capsule": (0.2588, 0.2575, 0.2366)
    }
    def __init__(
        self, datatype: str="bottle", root: str=ROOTDATADIR, train: bool=True, download: bool=True, 
        transforms: Union[tfms.Compose, List[tfms.Compose]]=tfms.Compose([tfms.ToTensor(),]), 
        dtype_data=torch.float32, dtype_target=torch.long, **kwargs
    ):
        assert isinstance(datatype, str) and datatype in [
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", 
            "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
        ]
        self.dirpath = correct_dirpath(root) + f"MVTecAD/{datatype}/"
        # download
        if   datatype == "bottle":     url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz"
        elif datatype == "cable":      url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz"
        elif datatype == "capsule":    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz"
        elif datatype == "carpet":     url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz"
        elif datatype == "grid":       url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz"
        elif datatype == "hazelnut":   url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz"
        elif datatype == "leather":    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz"
        elif datatype == "metal_nut":  url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz"
        elif datatype == "pill":       url = "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz"
        elif datatype == "screw":      url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz"
        elif datatype == "tile":       url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz"
        elif datatype == "toothbrush": url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz"
        elif datatype == "transistor": url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz"
        elif datatype == "wood":       url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz"
        elif datatype == "zipper":     url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"
        deploy_files(url, self.dirpath, download, extract="tar")
        # set data infomation
        df = None
        if train:
            df    = pd.DataFrame(get_file_list(self.dirpath + f"{datatype}/train/", regex_list=[r"\.png"]), columns=["filepath"])
        else:
            df    = pd.DataFrame(get_file_list(self.dirpath + f"{datatype}/test/",         regex_list=[r"\.png"]), columns=["filepath"])
            df_gt = pd.DataFrame(get_file_list(self.dirpath + f"{datatype}/ground_truth/", regex_list=[r"\.png"]), columns=["filepath_mask"])
        df["filename"]   = df["filepath"].apply(lambda x: os.path.basename(x))
        df["label_name"] = df["filepath"].str.split("/").str[-2]
        df["label"]      = df["label_name"].map({"good": 0}).fillna(1).astype(int)
        df["id"]         = df["filename"].str.replace(".png", "", regex=False)
        if train == False:
            df_gt["filename_mask"] = df_gt["filepath_mask"].apply(lambda x: os.path.basename(x))
            df_gt["label_name"]    = df_gt["filepath_mask"].str.split("/").str[-2]
            df_gt["id"]            = df_gt["filename_mask"].str.replace("_mask.png", "", regex=False)
            df = pd.merge(df, df_gt, how="left", on=["label_name", "id"])
        self.df = df
        # cretate dataset
        dataset  = ImageDataset(
            self.df["filepath"].values.tolist(), self.df["label"].values.tolist(), 
            transforms=transforms
        )
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)
