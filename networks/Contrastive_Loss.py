import sys
import torch
import torch.nn.functional as F


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    #p_i_j = F.cosine_similarity(view1.unsqueeze(2), view2.unsqueeze(1), dim=2)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def crossview_contrastive_Loss(view1, view2, lamb=0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    bn = view1.size(0)
    view1 = torch.reshape(view1, (bn, -1))    
    view2 = torch.reshape(view2, (bn, -1))       
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
    
#     Works with pytorch <= 1.2
#     p_i_j[(p_i_j < EPS).data] = EPS
#     p_j[(p_j < EPS).data] = EPS
#     p_i[(p_i < EPS).data] = EPS
    
    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device = p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device = p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device = p_i.device), p_i)

    loss =  p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()

    return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def twin_contrastive(lambd, z1, z2):
        
    bn = z1.size(0)
    # empirical cross-correlation matrix
    #z1 = torch.reshape(z1, (bn, -1))    
    #z2 = torch.reshape(z2, (bn, -1)) 
    
    #bnorm_size = z1.size(-1)
    #bnorm_layer = torch.nn.BatchNorm1d(bnorm_size, affine=False).to(z1.device)

    #z1 = bnorm_layer(z1)
    #z2 = bnorm_layer(z2)
    c = z1.T @ z2

    # sum the cross-correlation matrix between all gpus
    c.div_(bn)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag

    return loss

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    p_loss = p_loss.sum()

    return p_loss

def contrastive_info(z1, z2, tem = 1):
    bn = z1.size(0)
    z1 = torch.reshape(z1, (bn, -1))    
    z2 = torch.reshape(z2, (bn, -1))     
    Q_pos = F.cosine_similarity(z1, z2, dim=1)   #[1, 135]
    #single_in_log = torch.exp(Q_pos/tem)
    #loss = torch.sum(-1 * torch.log(single_in_log), dim=0) / bn 
    loss = torch.sum(Q_pos, dim=0) / bn 
    return -loss

def global_cosine(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1, 1]
    for item in range(len(a)):
        
        a_ = a[len(a) - 1 - item]
        b_ = b[item]
        #print(a_.size())
        #print(b_.size())
        loss += torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1),
                                        b_.view(b_.shape[0], -1))) * weight[item]
        #b_.register_hook(lambda grad: grad * 0)
    return loss