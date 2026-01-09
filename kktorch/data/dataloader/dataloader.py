from typing import Callable, List, Union, Dict
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from kktorch.util.com import check_type_list


__all__ = [
    "BaseDataLoader",
    "RandomDataLoader",
    "NumpyDataLoader",
]


class BaseDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, 
        dtype_data:   Union[torch.dtype, List[torch.dtype], Dict[object, torch.dtype]], 
        dtype_target: Union[torch.dtype, List[torch.dtype], Dict[object, torch.dtype]],
        **kwargs
    ):
        """
        Usage::
            BaseDataLoader(
                dataset, dtype_data=torch.float32, dtype_target=torch.long,
                batch_size=8, shuffle=True
            )
        """
        self.dtype_data   = dtype_data
        self.dtype_target = dtype_target
        if kwargs.get("collate_fn") is None: kwargs["collate_fn"] = self.collate_fn
        if kwargs.get("pin_memory") is None: kwargs["pin_memory"] = True
        super().__init__(dataset, **kwargs)
        self.is_check  = True
        self.procs_to_tensor_data = []
        self.proc_to_tensor_label = lambda x: x
        self.procs_to_dtype_data  = []
        self.proc_to_dtype_label  = lambda x, y: x.to(y)
    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return self.collate_fn((sample,))
    def __len__(self):
        return len(self.dataset)
    def collate_fn(self, batch):
        *input, label = list(zip(*batch))
        return self.to_tensor(input, label)
    @classmethod
    def judge_to_tensor_function(cls, input):
        error = Exception(f"Not match input.\ntype: {type(input)}\ninput: {input}")
        if isinstance(input, dict):
            return lambda x: {y:torch.Tensor(z) for y, z in x.items()}
        elif isinstance(input, list) or isinstance(input, tuple):
            if isinstance(input, tuple): input = list(input)
            if   check_type_list(input, [torch.Tensor]):
                return torch.stack
            elif check_type_list(input, [int, float], [int, float], [int, float], [int, float]):
                return torch.Tensor
            else:
                raise error
        elif isinstance(input, np.ndarray):
            return torch.from_numpy
        else:
            raise error
    @classmethod
    def judge_to_dtype_function(cls, input, dtype):
        error = Exception(f"Not match dtype: {dtype}")
        if isinstance(input, dict):
            if isinstance(dtype, dict):
                return lambda x, y: {a:b.to(y[a]) for a, b in x.items()}
            elif isinstance(dtype, torch.dtype):
                return lambda x, y: {a:b.to(y)    for a, b in x.items()}
            else:
                raise error
        elif isinstance(input, list) or isinstance(input, tuple):
            if isinstance(dtype, list) and check_type_list(dtype, torch.dtype):
                assert len(dtype) == len(input)
                return lambda x, y: [_input.to(y[i]) for i, _input in enumerate(x)]
            elif isinstance(dtype, torch.dtype):
                return lambda x, y: [_input.to(y) for _input in x]
            else:
                raise error
        else:
            return lambda x, y: x.to(y)
    def to_tensor(self, input: List[object], label):
        """
        Assume multiple inputs, so input is assumed to be lsit.
        The dataset i returns a value like the following.
            (tensor([0.7227, 0.1555]), tensor([0.7227, 0.1555]), [1,2])
                dataA_i: tensor([0.7227, 0.1555])
                dataB_i: tensor([0.7227, 0.1555])
                label_i: [labelx, labely]
        Receive this in batch (size N).
            >>> *input, label = list(zip(*batch))
            >>> input[0]
            [dataA_1, dataA_2, ..., dataA_N]
            >>> input[1]
            [dataB_1, dataB_2, ..., dataB_N]
            >>> label
            [label_1, label_2, ..., label_N]
        Then combine these lists and convert them to tensors.
            >>> torch.stack(input[0])
            tensor([[0.6719, 0.3023],
                    [0.6470, 0.2717],
                    ...,
                    [0.8315, 0.9313],
                    [0.9756, 0.9820]])
            >>> torch.stack(input[0]).shape
            torch.Size([N, 2]) # 2 is dimension.
        """
        if self.is_check:
            assert isinstance(input, list)
            procs_to_tensor_data, procs_to_dtype_data = [], []
            for _input in input:
                procs_to_tensor_data.append(self.judge_to_tensor_function(_input))
                output = procs_to_tensor_data[-1](_input)
                procs_to_dtype_data.append(self.judge_to_dtype_function(output, self.dtype_data))
            self.procs_to_tensor_data = lambda x   : [procs_to_tensor_data[i](_input   ) for i, _input in enumerate(x)]
            self.procs_to_dtype_data  = lambda x, y: [procs_to_dtype_data[ i](_input, y) for i, _input in enumerate(x)]
            self.proc_to_tensor_label = self.judge_to_tensor_function(label)
            self.proc_to_dtype_label  = self.judge_to_dtype_function(self.proc_to_tensor_label(label), self.dtype_target)
            self.is_check = False
        input = self.procs_to_tensor_data(input)
        input = self.procs_to_dtype_data( input, self.dtype_data)
        if len(input) == 1: input = input[0]
        label = self.proc_to_tensor_label(label)
        label = self.proc_to_dtype_label(label, self.dtype_target)
        return input, label


class TextDataLoader(BaseDataLoader):
    def __init__(
        self, dataset: Dataset, dtype_data: torch.dtype, dtype_target: torch.dtype, 
        tokenizer: Callable, tokenizer_params_input: dict={}, tokenizer_params_label: dict={}, 
        preprocs: List[Callable]=None, aftprocs: List[Callable]=None,
        **kwargs
    ):
        """
        Usually, it needs to be converted to Tensor in Dataset.
        In the case of Text, since it is more efficient to tokenize a list of sentences together, 
        DataLoader converts them into Tensors.
        Params::
            preprocs, aftprocs: process argument is "input", "label"
                ex) preprocs=[lambda x, y: [dict(x), y]]
        """
        super().__init__(dataset, dtype_data, dtype_target, **kwargs)
        self.tokenizer = tokenizer
        assert isinstance(tokenizer_params_input, dict)
        assert isinstance(tokenizer_params_label, dict)
        self.tokenizer_params_input = tokenizer_params_input
        self.tokenizer_params_label = tokenizer_params_label
        self.preprocs = preprocs if isinstance(preprocs, list) else ([preprocs, ] if preprocs is not None else [])
        self.aftprocs = aftprocs if isinstance(aftprocs, list) else ([aftprocs, ] if aftprocs is not None else [])
        for proc in self.preprocs: assert proc.__code__.co_nlocals == 2
        for proc in self.aftprocs: assert proc.__code__.co_nlocals == 2
    def collate_fn(self, batch):
        input, label = list(zip(*batch))
        input, label = list(input), list(label)
        for proc in self.preprocs: input, label = proc(input, label)
        input = self.tokenizer(input, **self.tokenizer_params_input)
        if check_type_list(label, str): label = self.tokenizer(label, **self.tokenizer_params_label)
        for proc in self.aftprocs: input, label = proc(input, label)
        return self.to_tensor([input,], label)


class RandomDataLoader(BaseDataLoader):
    def __init__(
        self, n_data: int, n_features: Union[int, List[int]], n_classes: int, 
        target_type: str="cls", **kwargs
    ):
        assert isinstance(target_type, str) and target_type in ["cls", "reg"]
        n_features   = n_features if isinstance(n_features, list) else [n_features, ]
        data         = torch.rand(n_data, *n_features)
        dtype_target = torch.long
        if target_type == "cls":
            label = torch.randint(n_classes, [n_data,])
        else:
            label = torch.rand(n_data, n_classes)
            dtype_target = torch.float32
        dataset = TensorDataset(data, label)
        super().__init__(dataset, dtype_data=torch.float32, dtype_target=dtype_target, **kwargs)


class NumpyDataLoader(BaseDataLoader):
    def __init__(
        self, data: np.ndarray, target: np.ndarray, 
        dtype_data=torch.float32, dtype_target=torch.long,
        **kwargs
    ):
        data   = torch.from_numpy(data)
        target = torch.from_numpy(target)
        dataset = TensorDataset(data, target)
        super().__init__(dataset, dtype_data=dtype_data, dtype_target=dtype_target, **kwargs)
