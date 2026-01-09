import re, copy
import numpy as np
import torch
from torch import nn
from typing import List, Union

# local packages
import kktorch.util.tensor as util
from kktorch.util.com import check_type_list


__all__ = [
    "BaseModule",
    "TorchModule",
    "MiddleSaveOutput",
    "MiddleReleaseOutput",
    "SplitOutput",
    "CombineListInput",
    "EinsumInput",
    "ReshapeInput",
    "SelectIndexListInput",
    "SelectIndexTensorInput",
    "ParameterModule",
    "SharedParameterModule",
    "AggregateInput",
    "EvalModule",
    "SharedVariablesModule",
    "PretrainedModule",
    "TimmModule",
    "HuggingfaceModule",
]


class BaseModule(nn.Module):
    def __init__(self, name: str=None, is_no_grad: bool=False):
        assert isinstance(is_no_grad, bool)
        super().__init__()
        self.name        = self.__class__.__name__ if name is None else name
        self.is_debug    = False
        self.save_output = None
        self.is_no_grad  = is_no_grad
    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        output = input
        if self.is_no_grad:
            with torch.no_grad():
                output = self.forward_child(output)
        else:
            output = self.forward_child(output)
        if self.is_debug:
            self.save_output = {self.name: output.clone() if isinstance(output, torch.Tensor) else output}
        return output
    def forward_child(self, input):
        return input


class TorchModule(BaseModule):
    def __init__(self, nn_name: str, nn_args=[], nn_kwargs: dict={}, input_type: str="", name: str=None, **kwargs):
        assert isinstance(input_type, str) and input_type in ["", "*", "**"]
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.nn = getattr(nn, nn_name)(*nn_args, **nn_kwargs)
        self._forward = lambda x, network: network(x)
        if   input_type == "*":  self._forward = lambda x, network: network( *x)
        elif input_type == "**": self._forward = lambda x, network: network(**x)
    def forward_child(self, input: torch.Tensor):
        return self._forward(input, self.nn)


class MiddleSaveOutput(BaseModule):
    def __init__(self, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.middle_output = None
    def forward_child(self, input: torch.Tensor):
        output = input
        self.middle_output = output.clone()
        return output


class MiddleReleaseOutput(BaseModule):
    def __init__(self, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
    def forward_child(self, input: torch.Tensor):
        output = input
        return output


class SplitOutput(BaseModule):
    def __init__(self, n_split: int, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        assert isinstance(n_split, int) and n_split > 1
        self.n_split = n_split
    def extra_repr(self):
        return f'n_split={self.n_split}'
    def forward_child(self, input: torch.Tensor):
        output = [input] + [input.clone() for _ in range(self.n_split - 1)]
        return output


class CombineListInput(BaseModule):
    def __init__(self, combine_type: str, dim: int=-1, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        assert isinstance(combine_type, str) and combine_type in ["sum", "ave", "cat"]
        self.is_check       = True
        self.dim            = dim
        self.forward_output = None
        self.combine_type   = combine_type
        if   combine_type == "sum": self.forward_output = self.forward_sum
        elif combine_type == "ave": self.forward_output = self.forward_ave
        elif combine_type == "cat": self.forward_output = self.forwart_cat
    def extra_repr(self):
        return f'combine_type={self.combine_type}, dim={self.dim}'
    def forward_child(self, input: List[torch.Tensor]) -> torch.Tensor:
        if self.is_check:
            # Check only the first time.
            assert isinstance(input, list) and len(input) > 0
            self.is_check = False
        output = self.forward_output(input)
        return output
    def forward_sum(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = input[0]
        for x in input[1:]: output += x
        return output
    def forward_ave(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = self.forward_sum(input)
        return output / len(input)
    def forwart_cat(self, input: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(input, dim=self.dim)


class SelectIndexListInput(BaseModule):
    def __init__(self, index: int, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.index = index
    def forward_child(self, input: List[torch.Tensor]) -> torch.Tensor:
        output = input[self.index]
        return output


class EinsumInput(BaseModule):
    def __init__(self, einsum: str, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.einsum = einsum
    def extra_repr(self):
        return f'einsum={self.einsum}'
    def forward_child(self, input: Union[torch.Tensor, List[torch.Tensor]]):
        output = input
        if not isinstance(input, list): output = [output]
        output = torch.einsum(self.einsum, *output)
        return output


class ReshapeInput(BaseModule):
    def __init__(self, *dim, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.dim        = dim 
        self.is_not_set = True if sum([isinstance(x, str) for x in dim]) > 0 else False
    def extra_repr(self):
        return f'dim={self.dim}'
    def forward_child(self, input: torch.Tensor):
        if self.is_not_set:
            self.dim = self.convert_dim(self.dim, input.shape)
            self.is_not_set = False
        output = input.reshape(*self.dim)
        return output
    @classmethod
    def convert_dim(cls, dim: list, shape: List[int]) -> List[int]:
        list_output = []
        dict_shape = {list("abcdefg")[i]:x for i, x in enumerate(shape)}
        for x in dim:
            if isinstance(x, str):
                list_output.append(eval(x, dict_shape))
            else: list_output.append(x)
        return list_output


class SelectIndexTensorInput(BaseModule):
    def __init__(
        self, max_dim: int, dim1=None, dim2=None, dim3=None, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"))
        assert max_dim >= 1 and max_dim < 7
        self.max_dim = max_dim
        self.dim1    = self.convert_dim(dim1)
        self.dim2    = self.convert_dim(dim2)
        self.dim3    = self.convert_dim(dim3)
        for x, y in kwargs.items(): setattr(self, x, self.convert_dim(y))
        self.forward_output = getattr(self, f"select_dim{max_dim}")
        for i in range(1, max_dim+1):
            if hasattr(self, f"dim{i}") == False:
                setattr(self, f"dim{i}", self.convert_dim(None))
    def extra_repr(self):
        return f'max_dim={self.max_dim}, dim={[getattr(self, "dim"+str(i)) for i in range(1, self.max_dim+1)]}'
    @classmethod
    def convert_dim(cls, dim: Union[int, str, List[int]]):
        if dim is None:
            return slice(None)
        elif isinstance(dim, int):
            return dim
        elif isinstance(dim, range):
            return list(dim)
        elif isinstance(dim, str):
            if dim.find(":") >= 0:
                return slice(*[int(x) if x != "None" else None for x in dim.split(":")])
            else:
                if dim.find("allpattern(") == 0:
                    x = int(eval(dim[len("allpattern("):-1], {}))
                    return torch.from_numpy(np.array([list(str(bin(i))[2:].zfill(int(np.log2(x - 1)) + 1)) for i in range(x)]).astype(int))
        elif isinstance(dim, list):
            return dim
    def select_dim1(self, input: torch.Tensor):
        return input[self.dim1]
    def select_dim2(self, input: torch.Tensor):
        return input[self.dim1, self.dim2]
    def select_dim3(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3]
    def select_dim4(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3, self.dim4]
    def select_dim5(self, input: torch.Tensor):
        return input[self.dim1, self.dim2, self.dim3, self.dim4, self.dim5]
    def forward_child(self, input: torch.Tensor):
        output = self.forward_output(input)
        return output


class ParameterModule(BaseModule):
    def __init__(
        self, *dim, 
        init_type: str="rand", init_timing: str="before", init_eval_str: str="",
        requires_grad: bool=True, dtype=torch.float32, output_type: str="parallel", 
        eps: float=1e-6, name: str="param", **kwargs
    ):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        assert not (init_timing == "after" and requires_grad)
        assert init_type in ["rand", "randn", "zero", "one", "custom", "eval"]
        self.dim            = dim
        self.init_type      = init_type
        self.init_timing    = init_timing
        self.init_eval_str  = init_eval_str
        self.requires_grad  = requires_grad
        self.dtype          = eval(dtype, {"torch": torch}) if isinstance(dtype, str) else dtype
        self.is_not_set     = True if self.init_timing == "after" else False 
        if self.init_timing == "before": self.create_parameter(*self.dim)
        self.output_type    = output_type
        self.forward_output = lambda x, y: x
        if   output_type == "parallel": self.forward_output = lambda x, y: [x, y]
        elif output_type == "plus":     self.forward_output = lambda x, y: x + y
        elif output_type == "minus":    self.forward_output = lambda x, y: x - y
        elif output_type == "times":    self.forward_output = lambda x, y: x * y
        elif output_type == "divide":   self.forward_output = lambda x, y: x / (y + eps)
        elif output_type == "exp":      self.forward_output = lambda x, y: x * torch.exp(y + eps)
        elif output_type == "log":      self.forward_output = lambda x, y: x * torch.log(y + eps)
        else:                           self.forward_output = lambda x, y: eval(self.output_type, {"input": x, "param": y, "torch": torch})
    def extra_repr(self):
        return  f'name={self.name}, dim={self.dim}, init_type={self.init_type}, init_timing={self.init_timing}, ' + \
                f'init_eval_str={self.init_eval_str}, requires_grad={self.requires_grad}, dtype={self.dtype}, output_type: {self.output_type}'
    def forward_child(self, input: torch.Tensor):
        if self.is_not_set:
            self.create_parameter(*self.convert_dim(self.dim, input.shape), device=input.device)
            self.is_not_set = False
        output = self.forward_output(input, self.param)
        return output
    def create_parameter(self, *dim, device="cpu"):
        param = None
        if   self.init_type == "rand":   param = torch.rand( *dim)
        elif self.init_type == "randn":  param = torch.randn( *dim)
        elif self.init_type == "zero":   param = torch.zeros(*dim)
        elif self.init_type == "one":    param = torch.ones( *dim)
        elif self.init_type == "custom": param = torch.Tensor(dim)
        elif self.init_type == "eval":   param = eval(self.init_eval_str, {"util": util, "init": torch.nn.init, "param": torch.empty(*dim)})
        self.param = nn.Parameter(param.to(self.dtype), requires_grad=self.requires_grad).to(device)
    @classmethod
    def convert_dim(cls, dim: list, shape: List[int]) -> List[int]:
        list_output = []
        for x in dim:
            if isinstance(x, str): list_output.append(shape[int(x)])
            else: list_output.append(x)
        return list_output


class SharedParameterModule(BaseModule):
    def __init__(
        self, param_address: str, name: str=None, is_copy: bool=False, requires_grad: bool=True, 
        output_eval: str="[x, y]", **kwargs
    ):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.param          = None
        self.param_address  = param_address
        self.is_copy        = is_copy
        self.requires_grad  = requires_grad
        self.output_eval    = output_eval
        self.forward_output = lambda x, y, output_eval, torch: eval(output_eval, {"input": x, "param": y, "torch": torch})
    def extra_repr(self):
        return f'name={self.name}, param_address={self.param_address}, is_copy={self.is_copy}, requires_grad={self.requires_grad}, output_eval: {self.output_eval}'
    def forward_child(self, input: torch.Tensor):
        output = self.forward_output(input, self.param, self.output_eval, torch)
        return output
    def set_parameter(self, param: nn.parameter.Parameter):
        if self.is_copy:
            param = copy.deepcopy(param)
            param.requires_grad = self.requires_grad
        self.param = param


class AggregateInput(BaseModule):
    def __init__(self, aggregate: str, name: str=None, is_no_grad: bool=False, **kwargs):
        """
        Params::
            aggregate: torch function.
                ex) sum, prod, mean, ...
        """
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), is_no_grad=is_no_grad)
        self.aggregate      = aggregate
        self.kwargs         = kwargs
        self.forward_output = getattr(torch, aggregate)
    def extra_repr(self):
        return f'aggregate={self.aggregate}, kwargs={self.kwargs}'
    def forward_child(self, input: torch.Tensor):
        output = self.forward_output(input, **self.kwargs)
        return output


class EvalModule(BaseModule):
    def __init__(self, eval: str, name: str=None, **kwargs):
        super().__init__(name=(self.__class__.__name__ if name is None else f"{self.__class__.__name__}({name})"), **kwargs)
        self.eval = eval
        self.variables = None
    def extra_repr(self):
        return f'eval={self.eval}'
    def forward_child(self, input: torch.Tensor):
        output = eval(self.eval, {"input": input, "torch": torch, "variables": self.variables})
        return output
    def set_variables(self, variables: dict):
        assert isinstance(variables, dict)
        self.variables = variables


class SharedVariablesModule(EvalModule):
    """
    https://github.com/kazukingh01/kktorch/blob/main/kktorch/model_zoo/swav/swav.json#L15
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_check = True
    def forward_child(self, input: torch.Tensor):
        variables = eval(self.eval, {"input": input, "torch": torch})
        if self.is_check:
            assert isinstance(variables, dict)
            self.is_check = False
        self.variables.update(variables)
        return input


class PretrainedModule(BaseModule):
    def __init__(
        self, model: nn.Module, name_model: str, name_module: str=None, 
        freeze_layers: Union[str, List[str]]=None, **kwargs
    ):
        """
        Params::
            name_module: If string, You can use a module that is part of a model named "name_module"
            dict_freeze:
                ex) {"Linear": 10}, Freeze all modules until "Linear" is encountered 10 times.
        """
        super().__init__(name=f"{self.__class__.__name__}({name_model})", **kwargs)
        self.name_model    = name_model
        self.freeze_layers = ([freeze_layers, ] if isinstance(freeze_layers, str) else freeze_layers) if freeze_layers is not None else []
        assert check_type_list(self.freeze_layers, str)
        self.model = model if name_module is None else getattr(model, name_module)
        self.freeze()
    def extra_repr(self):
        return f'name_model={self.name_model}'
    def freeze(self):
        for name, params in self.model.named_parameters():
            is_freeze = False
            for regstr in self.freeze_layers:
                if len(re.findall(regstr, name)) > 0:
                    is_freeze = True
                    break
            if is_freeze:
                params.requires_grad = False
            print(f"{name}: freeze ( {is_freeze} )")


class TimmModule(PretrainedModule):
    def __init__(self, name_model: str, pretrained: bool=True, set_ident_layers: List[str]=[], **kwargs):
        """
        Params::
            name_model: see: https://github.com/rwightman/pytorch-image-models/tree/master/timm/models
            pretrained: If true, load pretrained weight
        """
        assert isinstance(set_ident_layers, list) and check_type_list(set_ident_layers, str)
        import timm
        model = timm.create_model(name_model, pretrained=pretrained)
        for x in set_ident_layers:
            setattr(model, x, nn.Identity())
        super().__init__(model, name_model, **kwargs)
    def forward_child(self, input: torch.Tensor):
        output = self.model(input)
        return output


class HuggingfaceModule(PretrainedModule):
    def __init__(self, name_model: str, **kwargs):
        """
        Params::
            name_model: see: https://huggingface.co/models
        """
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained(name_model)
        super().__init__(model, name_model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.config    = model.config
        self.is_check  = True
    def forward_child(self, input: dict):
        if self.is_check:
            assert isinstance(input, dict)
            self.is_check = False
        output = self.model(**input)
        return output