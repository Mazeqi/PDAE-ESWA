from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset
from .base_net import BaseNet
import json
from networks.util import load_only_values
from networks.Model import Model
import os
from torch.optim import lr_scheduler
import torch
from kktorch.nn.loss import SSIMLoss
import logging
from networks.CDO_loss import CDO_LOSS
class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, trainer_para:dict, optim_para:dict, scheduler_para:dict, dataloader_para:dict, network_init_para:dict, out_path:str, mode:str):
        super().__init__()
        
        self.dataloader_para = dataloader_para
        self.trainer_para = trainer_para
        self.device = trainer_para['device']
        self.set_device()
        self.net_name = trainer_para['net_name']
        self.net = None
        self.mode = mode
        self.network_init_para = network_init_para
        self.loss_name = trainer_para['loss_name']

        if self.mode == "train":
            self.net = self.set_network(self.net_name, self.trainer_para, self.network_init_para, self.mode)

        if self.mode == "train":
            self.criterion = self.set_loss(self.loss_name)
            self.n_epochs = trainer_para['n_epochs']
            self.optimizer_name = trainer_para['optimizer_name']
            self.scheduler_name =trainer_para['scheduler_name']
            self.optim_para = optim_para
            self.schedule_para = scheduler_para
            self.optimizer = self.get_optimizer(self.net, self.optimizer_name, self.optim_para)
            self.scheduler = self.get_scheduler(self.optimizer, self.scheduler_name, self.schedule_para)

        if self.mode == "mv_train" or self.mode == "ov_train":
            self.criterion = self.set_loss(self.loss_name)
            self.n_epochs = trainer_para['n_epochs']
            self.optimizer_name = trainer_para['optimizer_name']
            self.scheduler_name =trainer_para['scheduler_name']
            self.optim_para = optim_para
            self.schedule_para = scheduler_para
        
        self.results = {
            'train_time': None,
            'test_time': None,
            'test_results': None,
        }
        
        self.out_path = out_path

    @abstractmethod
    def train(self, dataset) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass
    @abstractmethod
    def test(self, dataset: BaseADDataset):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass

    def set_device(self):
        # Default device to 'cpu' if cuda is not available
        if not torch.cuda.is_available():
            self.device = 'cpu'

    def save_results(self):
        """Save results dict to a JSON-file."""
        with open(self.out_path + '/result.json', 'w', encoding='utf8') as fp:
            json.dump(self.results, fp, indent=4)
            
    def set_network(self,  net_name, trainer_para, network_init_para, mode = "train"):
        logger = logging.getLogger()

        """Builds the neural network \phi."""
        if "in_ch" in trainer_para:
            model = Model("yamls/" + net_name + ".yaml", trainer_para["in_ch"], trainer_para["final_ch"], trainer_para["model_out_layers"])
        else:
            model = Model("yamls/" + net_name + ".yaml")
        if mode == "train":
            self.init_weights(model, **network_init_para["init_weight"])
        model.to(trainer_para["device"])
        logger.info(model)
        return model
    
    def set_loss(self, loss_name):
        if loss_name == "mse":
            criterion = torch.nn.MSELoss()
        elif loss_name == "ssim":
            criterion = SSIMLoss(8, 3)
        elif loss_name == "CDO":
            criterion = CDO_LOSS(OOM = True, gamma = self.trainer_para["CDO_GAMMA"])
        elif loss_name == "cos_sim":
            criterion = torch.nn.CosineSimilarity()
        else:
            criterion = torch.nn.MSELoss()
        return criterion
    def save_network(self, model, model_name):
        torch.save(model.state_dict(), self.out_path +"/" + model_name +".pth")
    
    def load_network(self, model, import_path):
        state_dict = torch.load(import_path)
        #if self.trainer_para["model_test_load_only_values"]:
        #    state_dict = load_only_values(state_dict, model.state_dict())
        model.load_state_dict(state_dict)
        return model

    def get_optimizer(self, model, optimizer_name, optim_para):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_para['adam'])
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **optim_para['sgd'])
        elif optimizer_name == "adamw":
           optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **optim_para['adamw'])
        else:
            return NotImplementedError('optimizer [%s] is not implemented', optimizer_name)

        return optimizer
        
        
    def get_scheduler(self, optimizer, scheduler_name, schedule_para):
        if scheduler_name == 'linear':
            def lambda_rule(epoch):
                epoch_count = self.schedule_para['linear']['inside_fun']['epoch_count']
                n_epochs_decay = self.schedule_para['linear']['inside_fun']['n_epochs_decay']
                lr_l = 1.0 - max(0, epoch + epoch_count  - self.n_epochs) / float(n_epochs_decay)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, **schedule_para["linear"]["linear_para"])
        elif scheduler_name == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, **schedule_para["step"])
        elif scheduler_name == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **schedule_para["plateau"])
        elif scheduler_name == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **schedule_para["cosine"])
        elif scheduler_name == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, **schedule_para["MultiStepLR"])
        elif scheduler_name == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer, **schedule_para["ExponentialLR"])
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', scheduler_name)
        return scheduler
    
    def init_weights(self, model, init_type='normal', init_gain=0.02):
        """ define the initialization function and batchnorms"""
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1: 
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        model.apply(init_func)  # apply the initialization function <init_func>