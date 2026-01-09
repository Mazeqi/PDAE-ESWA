import torch
import numpy as np
import random

def get_target_label_idx(labels, targets, pollution):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    if pollution >0 :
      poll = np.argwhere(~np.isin(labels,targets)).flatten().tolist()
      norm = np.argwhere(np.isin(labels, targets)).flatten().tolist()
      N = int(np.ceil(len(norm) * pollution))
      np.random.seed(1)
      ind = random.sample(range(0, len(poll)), N)
    
      poll = np.array(poll)[ind]

      
      ind = random.sample(range(0, len(norm)), len(norm) - N)
      norm = np.array(norm)[ind]
      index = poll.tolist() + norm.tolist()
    else:
      index = np.argwhere(np.isin(labels, targets)).flatten().tolist()
    return index


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x

