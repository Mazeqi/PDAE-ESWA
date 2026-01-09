import torch
import torch.nn as nn
import math

from networks.util import  PatchMaker

class Fusion_Layer(nn.Module):
    def __init__(self, fusion_method = "patch_cos", para = {}):
        super().__init__()

        self.fusion_method = fusion_method

        if self.fusion_method == "cos":
            self.criterion = torch.nn.CosineSimilarity()
        elif self.fusion_method == "patch_cos":
            if "fusion_size" in para:
                self.patch_make_score = PatchMaker(patchsize=para["fusion_size" ], dimension=512)
            else:
                self.patch_make_score = PatchMaker(patchsize=3, dimension=512)
            #self.patch_make_score.eval()
            self.criterion = torch.nn.CosineSimilarity()
        elif self.fusion_method == "patch_distance":
            self.patch_make_score = PatchMaker(patchsize=3, dimension=512)
            self.criterion = torch.nn.MSELoss()
    def forward(self, b1_fea, b2_fea):
        cos = 0

        if self.fusion_method == "cos":
            cos = torch.mean(1 - self.criterion(b1_fea.view(b1_fea.shape[0], -1), b2_fea.view(b2_fea.shape[0], -1)))
        elif self.fusion_method == "patch_cos":
            #with torch.no_grad():
            b1_fea = self.patch_make_score._embed(b1_fea, detech_fea = False)
            b1_fea = b1_fea.permute(0, 2, 1)
            b2_fea = self.patch_make_score._embed(b2_fea, detech_fea = False)
            b2_fea = b2_fea.permute(0, 2, 1)
            #print(b2_fea.shape)
            cos = torch.mean(1 - self.criterion(b1_fea.reshape(b1_fea.shape[0], -1), b2_fea.reshape(b2_fea.shape[0], -1)))
        elif self.fusion_method == "patch_distance":

            b1_fea = self.patch_make_score._embed(b1_fea, detech_fea = False)
            b2_fea = self.patch_make_score._embed(b2_fea, detech_fea = False)
            cos = self.criterion(b1_fea, b2_fea)
        return cos