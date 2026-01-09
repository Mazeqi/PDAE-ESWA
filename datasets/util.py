from .mvtec import MVTecDataset
from datasets.btad import BtadDataset
from datasets.mvtec_loco import MvtecLocoDataset
from datasets.mpdd import MpddDataset

from base.base_dataset import DatasetSplit

def load_dataset(dataset_name, data_path, image_size = 256, class_name = "bottle", task = None, dataset_para = {}):
    """Loads the dataset."""
    dataset = None
    """
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, pollution = pollution, N=N,normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class)

    if dataset_name == 'fmnist':
        dataset = FMNIST_Dataset(root=data_path, pollution = pollution, N=N, normal_class=normal_class)

    if dataset_name == 'mrart':
        dataset = MRART(root=data_path, pollution = pollution, N=N, normal_class=normal_class,seed=seed)
    """

    if dataset_name == 'mvtec':

        dataset = MVTecDataset(
                    source=data_path,
                    classname=class_name,
                    resize=image_size,
                    train_val_split=1,
                    split=task,
                    dataset_para = dataset_para
                )
    elif dataset_name == 'btad':
        dataset = BtadDataset(
                    source=data_path,
                    classname=class_name,
                    resize=image_size,
                    train_val_split=1,
                    split=task,
                    dataset_para = dataset_para
                )
    
    elif dataset_name == 'mvtec_loco':
        dataset = MvtecLocoDataset(
                    source=data_path,
                    classname=class_name,
                    resize=image_size,
                    train_val_split=1,
                    split=task,
                    dataset_para = dataset_para
                )
    elif dataset_name == 'mpdd':
        dataset = MpddDataset(
                    source=data_path,
                    classname=class_name,
                    resize=image_size,
                    train_val_split=1,
                    split=task,
                    dataset_para = dataset_para
                )
    return dataset
