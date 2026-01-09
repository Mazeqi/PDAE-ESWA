from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
class CDO_LOSS(nn.Module):
    def __init__(self, OOM = True, gamma = 2): 
        super(CDO_LOSS, self).__init__()

        self.OOM = OOM
        self.gamma = gamma
        
    def forward(self, fea_vec, fea_rec_vec, mask):

        B, C, H, W = fea_vec.shape
        Only_Normal = False
        if mask is not None:
            mask_vec = F.interpolate(mask, (H, W), mode='nearest')
        else:
            mask_vec = torch.zeros((B, 1, H, W))
            Only_Normal = True
        
        # reshape the mask, fe, and fa the the same shape for easily index
        mask_vec = mask_vec.permute(0, 2, 3, 1).reshape(-1, )

        fea_vec     = fea_vec.permute(0, 2, 3, 1).reshape(-1, C)
        fea_rec_vec = fea_rec_vec.permute(0, 2, 3, 1).reshape(-1, C)

        if Only_Normal:
            loss_n, weight_n = self.cal_discrepancy(fea_vec, fea_rec_vec, OOM=False, normal=True, gamma=1,
                                                aggregation=True)
            loss_s = 0
            weight_s = 0
        else:
            # normal features
            fe_n = fea_vec[mask_vec == 0]
            fa_n = fea_rec_vec[mask_vec == 0]

            # synthetic abnormal features
            fe_s = fea_vec[mask_vec != 0]
            fa_s = fea_rec_vec[mask_vec != 0]

            loss_n, weight_n = self.cal_discrepancy(fe_n, fa_n, OOM=self.OOM, normal=True, gamma=self.gamma,
                                                    aggregation=True)
            
            loss_s, weight_s = self.cal_discrepancy(fe_s, fa_s, OOM=self.OOM, normal=False, gamma=self.gamma,
                                                    aggregation=True)

        loss = ((loss_n + loss_s) / (weight_n + weight_s) * B)
        return loss

    def cal_discrepancy(self, fe, fa, OOM, normal, gamma=10, aggregation=True):

        # normalize the features into uint vector
        fe = F.normalize(fe, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)

        d_p = torch.sum((fe - fa) ** 2, dim=1)

        if OOM:
            # if OOM is utilized, we need to calculate the adaptive weights for individual features

            # calculate the mean discrepancy \mu_p to indicate the importance of individual features
            mu_p = torch.mean(d_p)

            if normal:
                # for normal samples: w = ((d_p) / \mu_p)^{\gamma}
                w = (d_p / mu_p) ** gamma

            else:
                # for abnormal samples: w = ((d_p) / \mu_p)^{-\gamma}
                w = (mu_p / d_p) ** gamma

            w = w.detach()

        else:
            # else, we manually assign each feature the same weight, i.e., 1
            w = torch.ones_like(d_p)

        if aggregation:
            d_p = torch.sum(d_p * w)

        sum_w = torch.sum(w)

        return d_p, sum_w
    
    @torch.no_grad()
    def cal_am(self, fea_vec, fea_rec_vec):
        

        fea_vec = F.normalize(fea_vec, p=2, dim=1)
        fea_rec_vec = F.normalize(fea_rec_vec, p=2, dim=1)
        
        remaps = torch.sum((fea_vec - fea_rec_vec) ** 2, dim=1).cpu().numpy() 
        scores = remaps.reshape(remaps.shape[0], -1).max(axis=1)

        return remaps, scores