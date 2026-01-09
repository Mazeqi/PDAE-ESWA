import torch
import torch
from torchvision import transforms
def rand_mask(img, mask_ratio = 0.3):
    ch, w, h = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape(ch, w*h)

    noise = torch.rand(1, w*h)

    # sort noise for each sample
    ratio = mask_ratio
    len_noise = int(ratio * (w*h))
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_noise = ids_shuffle[0, :len_noise]
    img[:, ids_noise] = noise[0, ids_noise]

    mask = torch.zeros([1, w*h])
    mask[:, ids_noise] = 1
    
    img = img.reshape(ch, w, h)
    mask = mask.reshape(1, w, h)
    return img, mask