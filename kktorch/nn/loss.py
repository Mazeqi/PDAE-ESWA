from typing import List
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


__all__ = [
    "IdentityLoss",
    "Accuracy",
    "CrossEntropyAcrossLoss",
    "CrossEntropySmoothingLoss",
    "SwAVLoss",
    "DINOLoss",
    "VAE_KLDLoss",
    "SSIMLoss",
    "DeepSVDDLoss",
]


class BaseLoss(_Loss):
    def __init__(self, reduction="mean", is_check_everytime: bool=True):
        assert isinstance(reduction, str) and reduction in ["mean", "sum", "ident"]
        super().__init__()
        self.is_check = True
        self.is_check_everytime = is_check_everytime
        self.output_reduction   = lambda x: x
        self.reduction          = reduction
        if   self.reduction == "mean": self.output_reduction = torch.mean
        elif self.reduction == "sum":  self.output_reduction = torch.sum
    def forward(self, *args, **kwargs):
        if self.is_check_everytime or self.is_check:
            self.check(*args, **kwargs)
            self.is_check = False
        return self.output_reduction(self.forward_child(*args, **kwargs))
    def check(self, *args, **kwargs): pass
    def forward_child(self):
        raise NotImplementedError


class IdentityLoss(BaseLoss):
    def __init__(self):
        super().__init__(reduction="ident", is_check_everytime=False)
    def forward_child(self, _, __):
        return 0


class Accuracy(BaseLoss):
    def __init__(self, threshold: float=0.5, is_check_everytime=False):
        super().__init__(reduction="mean", is_check_everytime=is_check_everytime)
        self.threshold = threshold
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if len(input.shape) == 1:
            assert len(target.shape) == 1
            assert input.shape[0] == target.shape[0]
        elif len(input.shape) == 2:
            assert target.dtype == torch.long
            assert len(target.shape) == 1
            assert input.shape[-1] >= torch.max(target)
    def forward_child(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 1:
            output = (target.to(torch.bool) == (input > self.threshold))
        elif len(input.shape) == 2:
            output = torch.max(input, dim=-1)[1]
            output = (output == target)
        output = output.to(torch.float32)
        return output


class CrossEntropyAcrossLoss(torch.nn.CrossEntropyLoss):
        def __init__(self, embedding_dim: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embedding_dim = embedding_dim
        def forward(self,input, target):
            return super().forward(input.reshape(-1, self.embedding_dim), target.reshape(-1))


class CrossEntropySmoothingLoss(BaseLoss):
    def __init__(self, classes: int, smoothing: float=0.0, reduction: str='mean', ignore_index: int=-100, is_check_everytime=False):
        assert isinstance(classes, int) and classes > 1
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.confidence   = 1.0 - smoothing
        self.smoothing    = smoothing
        self.classes      = classes
        self.ignore_index = ignore_index
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor) and len( input.shape) >= 2
        assert isinstance(target, torch.Tensor) and len(target.shape) == 1
    def forward_child(self, input: torch.Tensor, target: torch.Tensor):
        input = input.log_softmax(dim=-1)
        true_dist = None
        with torch.no_grad():
            true_dist = torch.zeros_like(input, requires_grad=False).to(input.device)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        output = torch.sum(-true_dist * input, dim=-1)
        output = output[target != self.ignore_index]
        return output


class SwAVLoss(BaseLoss):
    """
    see: https://arxiv.org/abs/2006.09882
    """
    def __init__(
        self, temperature: float=0.1, sinkhorn_epsilon=0.05, 
        sinkhorn_repeat: int=3, n_global_view: int=2,
        reduction: str='mean', is_check_everytime=False
    ):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.temperature      = temperature
        self.sinkhorn_repeat  = sinkhorn_repeat
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.n_global_view    = n_global_view
        self.log_softmax      = torch.nn.functional.log_softmax
    def check(self, input: torch.Tensor, *args):
        assert isinstance(input, torch.Tensor) and len(input.shape) == 3
    def forward_child(self, input: torch.Tensor, *args):
        """
        input: shape(Batch, N_Aug, K_cluster)
            # assume index starts from 1
            input[:,   1, :] ---> t1 Global Views
            input[:,   2, :] ---> t2 Global Views
            input[:,   3, :] ---> t3 Additional Small Views
            ...
            input[:, V+2, :] ---> tV+2 Additional Small Views
        """
        tens_zt = torch.einsum("abc->bac", input) # N_Aug, B_Batch, K_cluster
        with torch.no_grad():
            tens_zs  = tens_zt[:self.n_global_view].clone()
            tens_qs  = self.sinkhorn(tens_zs)
            tens_qs_list = []
            for i in range(self.n_global_view):
                tens_qs_list.append(tens_qs[i].expand(tens_zt.shape[0] - 1, -1, -1).detach())
        tens_pt = self.log_softmax(tens_zt / self.temperature, dim=2)
        losses  = [
            (-1 * tens_qs_list[i] * torch.cat([
                tens_pt[[j for j in range(self.n_global_view) if j != i]], tens_pt[self.n_global_view:]
            ], dim=0)).sum(dim=2) for i in range(self.n_global_view)
        ]
        loss = torch.cat([_loss.reshape(-1) for _loss in losses], dim=0)
        loss = loss / (self.n_global_view * (input.shape[0] - 1))
        return loss
    def sinkhorn(self, tens_zs: torch.Tensor):
        """
        tens_zs: shape(N_Aug, B_Batch, K_cluster)
        """
        Q = torch.exp(tens_zs / self.sinkhorn_epsilon).T
        Q = Q / torch.sum(Q, dim=(0,1))
        K, B, N = Q.shape
        u, r, c = torch.zeros(K, N), torch.ones(K, N) / K, torch.ones(B, N) / B
        u, r, c = u.to(tens_zs.device), r.to(tens_zs.device), c.to(tens_zs.device)
        for _ in range(self.sinkhorn_repeat):
            u = torch.sum(Q, dim=1)
            Q = Q * (r / u).unsqueeze(1)
            Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)
        Q = Q / torch.sum(Q, dim=0, keepdim=True)
        return Q.T


class DINOLoss(BaseLoss):
    """
    see: https://arxiv.org/abs/2104.14294
    """
    def __init__(
        self, temperature_s: float=0.1, temperature_t: float=0.04, 
        update_rate: float=0.9, n_global_view: int=2,
        reduction: str='mean', is_check_everytime=False
    ):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.temperature_s = temperature_s
        self.temperature_t = temperature_t
        self.vec_center    = None
        self.update_rate   = update_rate
        self.n_global_view = n_global_view
        self.softmax       = torch.nn.functional.softmax
        self.log_softmax   = torch.nn.functional.log_softmax
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input,  torch.Tensor) and len(input. shape) == 3
        assert isinstance(target, torch.Tensor) and len(target.shape) == 3
        assert input.shape == target.shape
        if self.vec_center is None:
            self.vec_center = torch.zeros(input.shape[-1], requires_grad=False).to(input.device)
    def forward_child(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: List of torch.Tensor, [tensor_student: torch.Tensor, tensor_teacher: torch.Tensor]
            shape(B_Batch, N_Aug, D_Dimension)
            # assume index starts from 1
            input[:,   1, :] ---> t1 Global Views
            input[:,   2, :] ---> t2 Global Views
            input[:,   3, :] ---> t3 Additional Small Views
            ...
            input[:, V+2, :] ---> tV+2 Additional Small Views
        """
        input  = input.permute(1,0,2) # N_Aug, B_Batch, D_Dimension
        with torch.no_grad():
            target        = target.detach()[:, :self.n_global_view, :].permute(1,0,2) # N_Aug, B_Batch, D_Dimension
            output_t      = self.softmax((target - self.vec_center) / self.temperature_t, dim=-1)
            output_t_list = [output_t[i].expand(input.shape[0] - 1, -1, -1) for i in range(self.n_global_view)]
        output_s = self.log_softmax(input / self.temperature_s, dim=-1)
        losses   = [
            (-1 * output_t_list[i] * torch.cat([
                output_s[[j for j in range(self.n_global_view) if j != i]], output_s[self.n_global_view:]
            ], dim=0)).sum(dim=2) for i in range(self.n_global_view)
        ]
        loss = torch.cat([_loss.reshape(-1) for _loss in losses], dim=0)
        loss = loss / (self.n_global_view * (input.shape[0] - 1))
        with torch.no_grad():
            self.vec_center = self.vec_center.mul(self.update_rate) + target.mean(dim=(0,1)).mul(1 - self.update_rate)
        return loss


class VAE_KLDLoss(BaseLoss):
    def __init__(self, reduction: str='mean', is_check_everytime=False):
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
    def check(self, input: torch.Tensor, target):
        assert isinstance(input, torch.Tensor) and len(input.shape) == 2
    def forward_child(self, input: torch.Tensor, target):
        """
        input::
            gaussian parameters.
            shape is (batch_size, z_mean_dim + z_sigma_dim)
            z_sigma_dim: log(sigma^2)
        target::
            None
        """
        dim = input.shape[1] // 2
        input_z_mean, input_z_logsigma2 = input[:, :dim], input[:, dim:]
        loss = -0.5 * (1 + input_z_logsigma2 - (input_z_mean ** 2) - torch.exp(input_z_logsigma2)).sum(axis=-1)
        return loss


class SSIMLoss(BaseLoss):
    def __init__(
        self, window_size: int, n_channel: int, sigma: float=1.5, padding: int=0,
        reduction: str='mean', is_check_everytime=False    
    ):
        """
        https://github.com/Po-Hsun-Su/pytorch-ssim
        """
        super().__init__(reduction=reduction, is_check_everytime=is_check_everytime)
        self.window_size  = window_size
        self.n_channel    = n_channel
        self.sigma        = sigma
        self.padding      = padding
        self.gauss_window = self.create_window(self.window_size, n_channel=self.n_channel, sigma=self.sigma)
        self.norm         = torch.nn.Hardtanh(0, 1) 
        L       = 1 # If 0 ~ 255 value range, L=255. If 0 ~ 1   value range, L=1.
        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2
        
    @classmethod
    def gaussian(cls, window_size: int, sigma: float=1.5):
        from math import exp
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    @classmethod
    def create_window(cls, window_size: int, n_channel: int=1, sigma: float=1.5):
        winsow_1d = SSIMLoss.gaussian(window_size, sigma).unsqueeze(1)
        window_2d = winsow_1d.mm(winsow_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window    = window_2d.expand(n_channel, 1, window_size, window_size).contiguous()
        return window
    def check(self, input: torch.Tensor, target: torch.Tensor):
        assert isinstance(input, torch.Tensor)  and len(input.shape)  == 4
        assert isinstance(target, torch.Tensor) and len(target.shape) == 4
        assert input.shape == target.shape
        #assert target.min().item() >= 0.0 and target.max().item() <= 1.0
        if self.is_check:
            self.gauss_window = self.gauss_window.to(input.device)
    def forward_child(self, input: torch.Tensor, target: torch.Tensor):
        """
        SSIM value is 0 ~ 1. value 1 means same image.
        """
        #input     = self.norm(input)
        mu_input  = F.conv2d(input,  self.gauss_window, padding=self.padding, groups=self.n_channel)
        mu_target = F.conv2d(target, self.gauss_window, padding=self.padding, groups=self.n_channel)
        mu_input_sq  = mu_input. pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_co        = mu_input * mu_target
        sigma_input_sq  = F.conv2d(input  * input,  self.gauss_window, padding=self.padding, groups=self.n_channel) - mu_input_sq
        sigma_target_sq = F.conv2d(target * target, self.gauss_window, padding=self.padding, groups=self.n_channel) - mu_target_sq
        sigma_co        = F.conv2d(input  * target, self.gauss_window, padding=self.padding, groups=self.n_channel) - mu_co
        numerator   = (2 * mu_co + self.C1) * (2 * sigma_co + self.C2)
        denominator = (mu_input_sq + mu_target_sq + self.C1) * (sigma_input_sq + sigma_target_sq + self.C2)
        loss        = numerator / denominator
        return 1 - loss.mean(axis=-1).mean(axis=-1).mean(axis=-1)


class DeepSVDDLoss(BaseLoss):
    def __init__(
        self, nu: float=0.1, objective: str="soft_boundary", n_update_C: int=0,
        is_check_everytime=False
    ):
        """
        Ref: https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py
        """
        assert isinstance(objective, str) and objective in ["soft_boundary", "hard"]
        assert isinstance(nu, float) and 0.0 <= nu and 1.0 > nu
        assert isinstance(n_update_C, int) and n_update_C >= 0
        super().__init__(reduction="ident", is_check_everytime=is_check_everytime)
        self.nu   = nu
        self.R    = 0 # radius R initialized with 0 by default. 
        self.C    = None
        self.iter = 0
        self.n_update_C = n_update_C
        self.is_soft    = True if objective == "soft_boundary" else False
    def check(self, input: torch.Tensor, *args, **kwargs):
        assert isinstance(input, torch.Tensor)
        assert len(input.shape) == 2
        if self.C is None:
            with torch.no_grad():
                self.C = torch.mean(input, dim=0)
    def forward_child(self, input: torch.Tensor, *args, **kwargs):
        dist = torch.sum((input - self.C) ** 2, dim=-1)
        if self.is_soft:
            with torch.no_grad():
                self.R = torch.quantile(torch.sqrt(dist), 1 - self.nu)
                if self.iter < self.n_update_C:
                    self.C = self.C + self.nu * (torch.mean(input, dim=0) - self.C) # center vector "C" will update little by little.
        if self.is_soft:
            scores = dist - (self.R ** 2)
            loss   = (self.R ** 2) + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        self.iter += 1
        return loss