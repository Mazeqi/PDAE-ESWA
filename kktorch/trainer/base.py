import os, datetime, random, copy, time
from typing import List, Union
import numpy as np

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
autocast = torch.cuda.amp.autocast

from kktorch.util.com import check_type_list, correct_dirpath, convert_1d_array, makedirs
from kktorch.data.dataloader import BaseDataLoader
from kktorch.util.logger import set_logger
logger = set_logger(__name__)


__all__ = [
    "Trainer",
    "EarlyStoppingError",
]


class EarlyStoppingError(Exception):
    """Early Stopping"""
    pass


class Trainer:
    def __init__(
        self,
        # network
        network: nn.Module,
        # loss functions
        losses_train: Union[_Loss, List[_Loss]]=None, losses_train_weight: List[float]=None, 
        losses_valid: Union[_Loss, List[_Loss], List[List[_Loss]]]=None,
        losses_train_name: Union[str, List[str]]=None, losses_valid_name: Union[str, List[str], List[List[str]]]=None, 
        adjust_output_size_front: int=0, adjust_target_size_front: int=0, 
        adjust_output_size_back:  int=0, adjust_target_size_back:  int=0, 
        # optimizer
        optimizer: dict={"optimizer": torch.optim.SGD, "params": dict(lr=0.001, weight_decay=0)},
        # scheduler
        scheduler: dict={"scheduler": None, "params": None, "warmup": None, "warmup_params": None},
        # dataloader
        dataloader_train: DataLoader=None, dataloader_valids: List[DataLoader]=[],
        # training parameter
        epoch: int=1, accumulation_step: int=1, clip_grad: float=0.0,
        # validation parameter
        valid_step: int=-1, valid_iter: int=1, early_stopping_rounds: int=-1, early_stopping_min_iter: int=-1, 
        move_ave_steps: int=1, early_stopping_i_valid: Union[int, List[int]]=None,
        # others
        outdir: str="./output_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S"), auto_mixed_precision: bool=False, 
        print_step: int=-1, save_step: int=None, random_seed: int=0
    ):
        """
        Params::
            network:
                network described by pytorch
            ### loss functions ###
            losses_train:
                training loss. The instance defined in "_Loss". 
                Require all losses to be backprobpagated.
                If a model has multiple outputs, define multiple outputs in list format.
                ex) 2 output
                    [Output1_Loss1, Output1_Loss2]
            losses_train_weight:
                If you want to add weights to multiple losses, define them here as a list.
            losses_valid:
                validation loss. There may be more than one evaluation metric, so if there are more than one, define them as follows
                ex) 2 output, 3 (2 + 1)  losses
                    losses_valid = [[Accuracy, AUC], [MSE]]
                    [[Output1_Accuracy, Output1_AUC], [Output2_MSE, ]]
            losses_train_name, losses_valid_name:
                The name to display in tensorboard. Does not have to be defined.
            ### optimizer ###
            optimizer:
                optimizer. Always write a class to define an instance internally.
            ### scheduler ###
            scheduler:
                scheduler. If not, don't define it. Always write a class to define an instance internally.
            ### dataloader ###
            dataloader_train, dataloader_valids:
                set dataloader. If there are multiple validations, define them in list.
            ### training parameter ###
            epoch:
                epoch.
            accumulation_step:
                Multiple batches of calculations can be performed before executing a single optimizer.step(), 
                which is useful when GPU memory is low and the batch size is small.
            ### validation parameter ###
            valid_step:
                The value of how many iters to calculate the validation.
                -1 is not calculated.
            early_stopping_rounds:
                The number of iterations to stop learning after the validation best parameter is updated.
            early_stopping_min_iter:
                Minimum number of iterations required for early stopping.
            move_ave_steps:
                Value of how many most recent validations should be used for averaging, in case validation cannot be calculated for the whole batch size.
            early_stopping_i_valid:
                index of which validation value to use.
                ex) 2 output, 3 (2 + 1) validation losses
                        [[Output1_Loss1, Output1_Loss2], [Output2_Loss3, ]]
                    If early_stopping_i_valid = 2,     it is used "Output2_Loss3" value for early stopping 
                    If early_stopping_i_valid = [1,2], it is used "Output1_Loss2" and "Output2_Loss3" sum value for early stopping
            ### others ###
            autocast:
                use mixed precision. torch.cuda.amp.autocast(enabled=True)
        """
        self.set_seed_all(random_seed)
        # network
        self.network = network
        # loss functions
        self.losses_train = losses_train if isinstance(losses_train, list) else ([losses_train]   if losses_train is not None else [])
        self.losses_valid = losses_valid if isinstance(losses_valid, list) else ([[losses_valid]] if losses_valid is not None else [])
        self.losses_train_weight = losses_train_weight if losses_train_weight is not None else 1.0
        self.losses_train_name = losses_train_name if isinstance(losses_train_name, list) else ([losses_train_name] if losses_train_name is not None else [])
        self.losses_valid_name = losses_valid_name if isinstance(losses_valid_name, list) else ([losses_valid_name] if losses_valid_name is not None else [])
        self.adjust_output_size_front = adjust_output_size_front
        self.adjust_output_size_back  = adjust_output_size_back
        self.adjust_target_size_front = adjust_target_size_front
        self.adjust_target_size_back  = adjust_target_size_back
        # optimizer
        self.optimizer_config = optimizer
        # scheduler
        self.scheduler_config = scheduler
        # DataLoader
        self.dataloader_train  = dataloader_train
        self.dataloader_valids = dataloader_valids if isinstance(dataloader_valids, list) else ([dataloader_valids] if dataloader_valids is not None else [])
        # training
        self.epoch             = epoch
        self.accumulation_step = accumulation_step
        self.clip_grad         = clip_grad
        # validation
        self.valid_step              = valid_step
        self.valid_iter              = valid_iter
        self.early_stopping_rounds   = early_stopping_rounds
        self.early_stopping_min_iter = early_stopping_min_iter
        self.early_stopping_i_valid  = early_stopping_i_valid
        self.move_ave_steps          = move_ave_steps
        # Config
        self.is_cuda = False
        # Other
        self.outdir     = correct_dirpath(outdir)
        self.save_step  = save_step
        self.print_step = print_step
        self.auto_mixed_precision = auto_mixed_precision
        # TensorBoard
        self.writer = None
        # Init
        self.initialize()
        # Check
        self.check_init()
        logger.info(f"{self}", color=["BOLD", "GREEN"])

    def __str__(self):
        string = f"""
network : \n{self.network}
loss functions train : \n{self.losses_train}
loss functions valid : \n{self.losses_valid}
optimizer : \n{self.optimizer}
epoch : {self.epoch}
"""
        return string
    
    def set_seed_all(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = True # If true, Faster processing. see: https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587
        logger.info('Set random seeds')
    
    def setup_optimizer(self):
        """
        Setup optimizer
        Format::
            config = {
                "optimizer": torch.optim.SGD, 
                "params": {"lr":0.001, "weight_decay":0}
            }
        """
        self.optimizer = None
        config = self.optimizer_config
        if isinstance(config, dict) and config.get("optimizer") is not None:
            self.optimizer = config.get("optimizer")(self.network.parameters(), **(config.get("params") if config.get("params") is not None else {}))
            assert isinstance(self.optimizer, Optimizer)

    def setup_scheduler(self):
        """
        Setup scheduler
        ## if you want to use "warmup scheduler", see "https://github.com/ildoonet/pytorch-gradual-warmup-lr"
        Format::
            config = {
                "scheduler": torch.optim.lr_scheduler.StepLR, 
                "params": {"step_size": 10, "gamma": 0.9},
                "warmup": warmup_scheduler.GradualWarmupScheduler,
                "warmup_params": {"multiplier": 1, "total_epoch": 5, "after_scheduler": "_scheduler"}
            }
        """
        self.scheduler = None
        assert isinstance(self.optimizer, Optimizer)
        config = self.scheduler_config
        if isinstance(config, dict) and config.get("scheduler") is not None:
            self.scheduler = config.get("scheduler")(self.optimizer, **(config.get("params") if config.get("params") is not None else {}))
            assert isinstance(self.scheduler, _LRScheduler)
            if config.get("warmup") is not None:
                warmup_params = config.get("warmup_params") if config.get("warmup_params") is not None else {}
                for x, y in warmup_params.items():
                    if isinstance(y, str) and y == "_scheduler": warmup_params[x] = self.scheduler
                self.scheduler = config.get("warmup")(self.optimizer, **warmup_params)
                assert isinstance(self.scheduler, _LRScheduler)

    def check_init(self):
        assert check_type_list(self.losses_train, _Loss)
        assert check_type_list(self.losses_valid, _Loss) or check_type_list(self.losses_valid, [list, _Loss], _Loss)
        if isinstance(self.losses_train_weight, list):
            assert check_type_list(self.losses_train_weight, float)
            assert len(self.losses_train_weight) == len(self.losses_train)
        else:
            assert isinstance(self.losses_train_weight, float)
        if len(self.losses_train_name) > 0: assert len(self.losses_train_name  ) == len(self.losses_train)
        if len(self.losses_valid_name) > 0:
            assert check_type_list(self.losses_valid_name, str)
            assert len(convert_1d_array(self.losses_valid_name)) == len(convert_1d_array(self.losses_valid))
        assert self.dataloader_train is None or isinstance(self.dataloader_train, DataLoader)
        assert check_type_list(self.dataloader_valids, DataLoader)
        assert isinstance(self.accumulation_step, int) and self.accumulation_step >= 1
        if isinstance(self.early_stopping_i_valid, int): self.early_stopping_i_valid = [self.early_stopping_i_valid]
        if self.early_stopping_i_valid is not None:
            assert check_type_list(self.early_stopping_i_valid, int)
        assert isinstance(self.valid_iter, int) and self.valid_iter >= -1
        assert isinstance(self.auto_mixed_precision, bool)
        assert isinstance(self.clip_grad, float) and self.clip_grad >= 0
        assert isinstance(self.adjust_output_size_front, int) and self.adjust_output_size_front >= 0
        assert isinstance(self.adjust_output_size_back,  int) and self.adjust_output_size_back  >= 0
        assert isinstance(self.adjust_target_size_front, int) and self.adjust_target_size_front >= 0
        assert isinstance(self.adjust_target_size_back,  int) and self.adjust_target_size_back  >= 0

    def initialize(self):
        logger.info("trainer parameter initialize.")
        self.iter       = 0
        self.iter_best  = 0
        self.i_epoch    = 0
        self.classes_   = None
        self.gpu_device = torch.device("cuda:0")
        self.early_stopping_iter = 0
        self.min_loss_valid  = float("inf")
        self.loss_valid_hist = np.full(self.move_ave_steps, float("inf")).astype(np.float32)
        self.best_params     = {"iter": 0, "loss_valid": float("inf"), "params": {}}
        self.setup_optimizer()
        self.setup_scheduler()
        self.scaler    = torch.cuda.amp.GradScaler()
        self.time_iter = 0

    @classmethod
    def _reset_parameters(cls, module: nn.Module, weight: Union[str, float]=None):
        try: module.reset_parameters()
        except AttributeError: pass
        if weight is None: pass
        elif isinstance(weight, str) and weight == "norm":
            if type(module) in [torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d]:
                if hasattr(module, "weight") and isinstance(module.weight, torch.nn.parameter.Parameter):
                    torch.nn.utils.weight_norm(module, "weight")
                if hasattr(module, "bias") and isinstance(module.bias, torch.nn.parameter.Parameter):
                    torch.nn.utils.weight_norm(module, "bias")
        else:
            try: module.weight.data.fill_(weight)
            except AttributeError: pass
            try: module.bias.data.fill_(weight)
            except AttributeError: pass

    def reset_parameters(self, weight: Union[str, float]=None):
        """
        Params::
            weight:
                float or str. If None, initialize with reset_parameters()
                If "norm", initialize with weight normalization.
        """
        for name, module in self.network.named_modules():
            logger.info(f"reset weights: {name}")
            self._reset_parameters(module, weight=weight)

    def to_cuda(self):
        self.network.to(self.gpu_device) # You can check whether the module is on cuda or not with "next(model.parameters()).is_cuda"
        def work(loss_funcs):
            funcs = [
                [(y.to(self.gpu_device) if hasattr(y, "to") else y) for y in x] if (isinstance(x, list) or isinstance(x, tuple)) \
                else (x.to(self.gpu_device) if hasattr(x, "to") else x) for x in loss_funcs
            ] 
            return funcs
        self.losses_train = work(self.losses_train) 
        self.losses_valid = work(self.losses_valid)
        self.is_cuda = True

    @classmethod
    def val_to_any(cls, input: Union[dict, list, tuple, torch.Tensor], anytype):
        if   isinstance(input, torch.Tensor):
            return input.to(anytype)
        elif isinstance(input, list) or isinstance(input, tuple):
            return [x.to(anytype) for x in input]
        elif isinstance(input, dict):
            return {x:y.to(anytype) if hasattr(y, "to") else y for x, y in input.items()}
        else:
            logger.raise_error(f"input value is not expected type. {input}")
    
    @classmethod
    def val_to_cpu(cls, input: Union[dict, list, tuple, torch.Tensor]):
        def work(_input: torch.Tensor):
            if _input.is_cuda:
                _input = _input.detach().to("cpu")
                try: return _input.item()
                except ValueError: return _input
            else:
                _input = _input.detach()
                try: return _input.item()
                except ValueError: return _input
        if isinstance(input, torch.Tensor):
            return work(input)
        elif isinstance(input, list) or isinstance(input, tuple):
            return [work(x) if isinstance(x, torch.Tensor) else x for x in input]
        elif isinstance(input, dict):
            return {x:work(y) if isinstance(y, torch.Tensor) else y for x, y in input.items()}

    def val_to_gpu(self, input: Union[dict, list, tuple, torch.Tensor]):
        if self.is_cuda:
            return self.val_to_any(input, self.gpu_device)
        return input
    
    def save(self, filename: str=None, is_best: bool=False):
        logger.info("model weight saving...", color="GREEN")
        if is_best:
            if self.best_params["iter"] > 0:
                torch.save(self.best_params["params"], self.outdir + filename \
                if filename is not None else self.outdir + f'model_best_{self.best_params["iter"]}.pth')
            else:
                logger.warning("self.best_params is nothing.")
                torch.save(self.network.state_dict(), self.outdir + filename + f".{self.iter}" if filename is not None else self.outdir + f"model_{self.iter}.pth")
        else:
            torch.save(self.network.state_dict(), self.outdir + filename + f".{self.iter}" if filename is not None else self.outdir + f"model_{self.iter}.pth")
    
    def load(self, model_path: str=None, is_best: bool=False):
        if is_best:
            model_path = self.outdir + f'model_best_{self.best_params["iter"]}.pth'
        logger.info(f"load weight: {model_path} start.")
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()
        logger.info(f"load weight: {model_path} end.")
    
    def process_data_train_pre(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        return input
    def process_data_train_aft(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(input, tuple): input = list(input)
        input = [input, ] if isinstance(input, torch.Tensor) else input
        return [torch.zeros(0)] * self.adjust_output_size_front + input + [torch.zeros(0)] * self.adjust_output_size_back
    def process_data_valid_pre(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        return input
    def process_data_valid_aft(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(input, tuple): input = list(input)
        input = [input, ] if isinstance(input, torch.Tensor) else input
        return [torch.zeros(0)] * self.adjust_output_size_front + input + [torch.zeros(0)] * self.adjust_output_size_back
    def process_label_pre(self, label: Union[torch.Tensor, List[torch.Tensor]], input: Union[torch.Tensor, List[torch.Tensor]]=None):
        return label
    def process_label_aft(self, label: Union[torch.Tensor, List[torch.Tensor]], input: Union[torch.Tensor, List[torch.Tensor]]=None):
        label = [label, ] if isinstance(label, torch.Tensor) else label
        return [torch.zeros(0)] * self.adjust_target_size_front + label + [torch.zeros(0)] * self.adjust_target_size_back

    def processes(
        self, 
        input: Union[torch.Tensor, List[torch.Tensor]], 
        label: Union[torch.Tensor, List[torch.Tensor]]=None, 
        is_valid: bool=False
    ) -> (Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]):
        """
        Process the output of dataloader up to the point before loss calculation.
        By default, the output will always be in list format. This is to account for multiple outputs and multiple loss calculations.
        """
        output = None
        # set pre/after proc based on whether it is training or not.
        proc_pre = self.process_data_valid_pre if is_valid else self.process_data_train_pre
        proc_aft = self.process_data_valid_aft if is_valid else self.process_data_train_aft
        # label pre proc ##Do not put val_to_gpu after network(input).It slows things down for some reason.
        if label is not None:
            label = self.val_to_gpu(self.process_label_pre(label, input=input))
        # input pre proc
        output = self.val_to_gpu(proc_pre(input))
        with autocast(enabled=self.auto_mixed_precision):
            output = self.network(output)
        # label after proc
        if label is not None:
            label = self.val_to_gpu(self.process_label_aft(label, input=output))
        # input after proc
        output = proc_aft(output)
        return output, label
    
    def calc_losses(
        self, 
        input: Union[torch.Tensor, List[torch.Tensor]], 
        label: Union[torch.Tensor, List[torch.Tensor]],
        is_valid: bool=False
    ) -> (float, List[float]):
        """
        Calculate the loss. Processing of "self.processes()" is also done internally.
        """
        def work(self, input, label, processes, loss_funcs, loss_funcs_weight=1.0, is_valid=False):
            output, label = processes(input, label=label, is_valid=is_valid)
            if self.print_step > 0 and (self.iter - 1) % self.print_step == 0:
                logger.info(f'iter: {self.i_epoch}|{self.iter}.\nSample output: \n{output}\nSample output label: \n{label}')
            # loss calculation
            loss, losses = 0, []
            with autocast(enabled=self.auto_mixed_precision):
                for i, loss_func in enumerate(loss_funcs):
                    if isinstance(loss_func, list):
                        # In validation, multiple evaluations can be performed on a single output.
                        for _loss_func in loss_func:
                            losses.append(_loss_func(output[i], label[i]))
                            loss = loss + losses[-1]
                    else:
                        loss_weight = loss_funcs_weight if isinstance(loss_funcs_weight, float) else loss_funcs_weight[i]
                        losses.append(loss_weight * loss_func(output[i], label[i]))
                        loss = loss + losses[-1]
            return loss, losses
        loss, losses = 0, []
        if is_valid:
            with torch.no_grad():
                loss, losses = work(self, input, label, self.processes, self.losses_valid, is_valid=is_valid)
        else:
            loss, losses = work(self, input, label, self.processes, self.losses_train, loss_funcs_weight=self.losses_train_weight, is_valid=is_valid)
        return loss, losses
    
    def write_tensor_board(self, name: str, value):
        if hasattr(value, "to"):
            value = value.detach().to("cpu").item()
        self.writer.add_scalar(name, value, global_step=self.iter)
    
    def preproc_update_weight(self): pass
    def aftproc_update_weight(self): pass
    
    def _train_step(self, input: Union[torch.Tensor, List[torch.Tensor]], label: Union[torch.Tensor, List[torch.Tensor]]):
        self.iter += 1
        self.network.train() # train() and eval() need to be distinguished when Dropout Layer is present
        if (self.iter - 1) % self.accumulation_step == 0:
            self.network.zero_grad()
            self.preproc_update_weight()
        if self.print_step > 0 and (self.iter - 1) % self.print_step == 0:
            logger.info(f"iter: {self.i_epoch}|{self.iter}.\nSample input: \n{input}\nSample input shape: \n{input.shape if isinstance(input, torch.Tensor) else ''}\nSample input label: \n{label}")
        loss, losses = self.calc_losses(input, label, is_valid=False)
        loss = loss / self.accumulation_step
        self.scaler.scale(loss).backward()
        if hasattr(self.optimizer, "first_step"):
            # For SUM optimizers. you can NOT use accumulation step.
            self.accumulation_step = 1
            self.optimizer.first_step(zero_grad=True)
            loss, losses = self.calc_losses(input, label, is_valid=False)
            self.scaler.scale(loss).backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            if self.iter % self.accumulation_step == 0:
                if self.accumulation_step > 1: logger.info("optimizer step with accumulation.")
                if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.aftproc_update_weight()
        if self.scheduler is not None: self.scheduler.step()
        loss, losses = self.val_to_cpu(loss), self.val_to_cpu(losses)
        logger.info(f'iter: {self.i_epoch}|{self.iter}, train: {loss}, losses: {losses}, time: {(time.perf_counter() - self.time_iter)}, lr: {"No schedule." if self.scheduler is None else self.scheduler.get_last_lr()[0]}')
        self.time_iter = time.perf_counter()
        # tensor board
        self.write_tensor_board("learning_rate", self.optimizer.defaults["lr"] if self.scheduler is None else self.scheduler.get_last_lr()[0])
        self.write_tensor_board("train/total_loss", loss)
        for i_loss, _loss in enumerate(losses):
            name = self.losses_train_name[i_loss] if len(self.losses_train_name) > 0 else f"loss_{i_loss}"
            self.write_tensor_board(f"train/{name}", _loss)
        # save
        if self.save_step is not None and self.save_step > 0 and self.iter % self.save_step == 0:
            self.save()
            self.save(is_best=True)

    def _valid_step(self, _input: List[object], label: List[object], i_valid: int=0):
        self.network.eval()
        with torch.no_grad():
            loss_valid, losses_valid, batch_size = 0, None, 0
            for i_batch in range(len(_input)):
                # loss calculation
                _size       = len(_input[i_batch])
                batch_size += _size
                _loss_valid, _losses_valid = self.calc_losses(_input[i_batch], label[i_batch], is_valid=True)
                _loss_valid, _losses_valid = self.val_to_cpu(_loss_valid), self.val_to_cpu(_losses_valid)
                loss_valid += _loss_valid * _size
                if losses_valid is None:
                    losses_valid = [x * _size for x in _losses_valid]
                else:
                    losses_valid = [losses_valid[i] + (x * _size) for i, x in enumerate(_losses_valid)]
            loss_valid, losses_valid = loss_valid / batch_size, [x / batch_size for x in losses_valid]
            if i_valid == 0:
                _loss_save = loss_valid if self.early_stopping_i_valid is None else np.sum(np.array(losses_valid)[self.early_stopping_i_valid])
                self.loss_valid_hist[self.iter // self.valid_step % self.move_ave_steps] = _loss_save
            # tensor board
            self.write_tensor_board(f"validation{i_valid}/total_loss", loss_valid)
            for i_loss, _loss in enumerate(losses_valid):
                name = self.losses_valid_name[i_loss] if len(self.losses_valid_name) > 0 else f"loss_{i_loss}"
                self.write_tensor_board(f"validation{i_valid}/{name}", _loss)
            self.early_stopping_iter += 1
            # early stopping conditions
            bool_store_early_stopping = False
            if i_valid == 0 and self.loss_valid_hist.max() < float("inf") and self.min_loss_valid > self.loss_valid_hist.mean():
                bool_store_early_stopping = True
            if bool_store_early_stopping:
                self.min_loss_valid = self.loss_valid_hist.mean()
                self.early_stopping_iter = 0 # iteration を reset
                self.best_params = {
                    "iter": self.iter,
                    "loss_valid": self.loss_valid_hist.mean(),
                    "params": copy.deepcopy(self.network.state_dict()),
                }
            logger.info(
                f'iter: {self.i_epoch}|{self.iter}, valid: {loss_valid}, losses: {losses_valid}, loss ave: {self.loss_valid_hist.mean()}, ' + \
                f'best iter: {self.best_params["iter"]}, best loss: {self.best_params["loss_valid"]}'
            )
            if isinstance(self.early_stopping_rounds, int) and self.early_stopping_rounds > 0 and self.early_stopping_iter >= self.early_stopping_rounds and \
               (self.early_stopping_min_iter is None or self.early_stopping_min_iter < 0 or self.iter > self.early_stopping_min_iter):
                # early stopping
                raise EarlyStoppingError

    def train(self):
        self.init_training()
        dataloader_valids = [iter(x) if isinstance(x, BaseDataLoader) else x for x in self.dataloader_valids]
        try:
            for i_epoch in range(self.epoch):
                self.i_epoch = i_epoch + 1
                for input, label in self.dataloader_train:
                    # train
                    self._train_step(input, label)
                    # validation
                    if len(dataloader_valids) > 0 and self.valid_step is not None and self.valid_step > 0 and self.iter % self.valid_step == 0:
                        for i_valid, dataloader_valid in enumerate(dataloader_valids):
                            input, label = [], []
                            if self.valid_iter >= 0:
                                for _ in range(self.valid_iter):
                                    try: _input, _label = next(dataloader_valid)
                                    except StopIteration:
                                        dataloader_valids[i_valid] = iter(self.dataloader_valids[i_valid])
                                        _input, _label = next(dataloader_valids[i_valid])
                                    input.append(_input)
                                    label.append(_label)
                            else:
                                for _input, _label in self.dataloader_valids[i_valid]:
                                    input.append(_input)
                                    label.append(_label)
                            self._valid_step(input, label, i_valid=i_valid)
        except EarlyStoppingError:
            logger.warning(f'early stopping. iter: {self.i_epoch}|{self.iter}, best_iter: {self.best_params["iter"]}, loss: {self.best_params["loss_valid"]}')
            self.iter_best = self.best_params["iter"]
        self.writer.close()
        self.save(is_best=True)
        self.save()
    
    def init_training(self):
        makedirs(self.outdir, exist_ok=True, remake=True)
        self.writer = SummaryWriter(log_dir=self.outdir + "logs")
        self.network.zero_grad()

    def predict(self, dataloader: DataLoader, is_label: bool=False, sample_size: int=-1):
        self.network.eval()
        output, label = None, None
        for i_batch, (_input, _label) in enumerate(dataloader):
            with torch.no_grad():
                _output, _label = self.processes(_input, label=_label if is_label else None, is_valid=True)
            if output is None:
                output = []
                for _ in range(len(_output)): output.append([])
            for i, x in enumerate(_output):
                output[i].append(self.val_to_cpu(x).numpy())
            if is_label:
                if label is None:
                    label = []
                    for _ in range(len(_label)): label.append([])
                for i, x in enumerate(_label):
                    label[i].append(self.val_to_cpu(x).numpy())
            if sample_size > 0 and (i_batch + 1) >= sample_size: break
        output = [np.concatenate(x, axis=0) for x in output]
        if len(output) == 1: output = output[0]
        if is_label:
            label = [np.concatenate(x, axis=0) for x in label]
            if len(label) == 1: label = label[0]
        return output, label
