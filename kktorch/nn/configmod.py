import os, json, copy, time
import torch
from kktorch import nn
from typing import Union, List
from kktorch.util.com import replace_sp_str_and_eval


__all__ = [
    "ConfigModule",
    "RepeatModule",
    "SplitModule",
    "ApplyModule",
    "RegidualModule",
]


class ConfigModule(nn.Module):
    def __init__(self, config: Union[str, dict], in_features: int=None, dir_base_path: str="./", user_parameters: dict=None, is_call_first: bool=True):
        """
        Params::
            config: string(json file path) or dictionary format.
            in_features: input n_features parameter. This parameter is in turn passed on to "__before".
            dir_base_path:
                You don't need to be particularly conscious of this when declaring it.
                Parameters for accessing child json in the same hierarchy as the parent json.
            user_parameters: Overwrite the user_parameters in the json file with the contents of this parameter
            is_call_first: No need to be particularly conscious of it.
        Format::
            {
                "name": "test", # network name. 
                "network": [
                    {
                        "class": "Linear",         # class name of torch.nn or kktorch.lib.modules
                        "args": [128, 64],         # List. input parameter of "class", ex) Linear(*args, **kwargs)
                        "kwargs": {"bias": False}, # Dict. input parameter of "class", ex) Linear(*args, **kwargs)
                        "out_features": "out_features"  # If string, attribute name for getting number of output nodes
                    },
                    {
                        "class": "ReLU"
                    },
                    {
                        "class": "ConfigModule",
                        "args": "sub.json" # Load sub.json in the same hierarchy
                    },
                    {
                        "class": "Linear",
                        "args": ["__before", 32], # "__before" is spetial word. The number of output nodes of the previous layer will be inherited.
                        "kwargs": {},
                        "out_features": 32 # If integer, this humber set to output nodes.
                    },
                ]
            }
        Usage::
            >>> import torch
            >>> from kktorch.nn import ConfigModule
            >>> network = ConfigModule({"name": "test", "network": [{"class": "Linear", "args": [128, 8]}, {"class": "ReLU"}]})
            >>> network
            ConfigModule(
                (test): ModuleList(
                    (0): Linear(in_features=128, out_features=8, bias=True)
                    (1): ReLU()
                )
            )
            >>> network(torch.rand(2, 128))
            tensor([[0.0000, 0.0621, 0.0000, 0.4308, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0377, 0.0000, 0.1924, 0.0000, 0.0000, 0.0788, 0.0000]],
                    grad_fn=<ReluBackward0>)
        """
        super().__init__()
        self.is_call_first = is_call_first
        fname            = (("" if self.is_call_first else dir_base_path) + config) if isinstance(config, str) and config[-5:] == ".json" else None
        self.config      = json.load(open(fname)) if fname is not None else config
        self.name        = self.config["name"]
        self.dirpath     = os.path.dirname(fname) + "/" if fname is not None else dir_base_path
        self.user_params = self.config.get("user_parameters") if self.config.get("user_parameters") is not None else {}
        ##  in_features priority is function Arguments "in_features" > config paramter "in_features" > 0
        in_features      = (self.config.get("in_features") if self.config.get("in_features") is not None else 0) if in_features is None else in_features
        self.user_params["in_features"] = in_features
        self.user_params["__before"]    = in_features
        self.user_params["__dirpath"]   = self.dirpath
        self.shared_variables = {}
        if user_parameters is not None and isinstance(user_parameters, dict):
            for x, y in user_parameters.items(): self.user_params[x] = y
        self.user_params = replace_sp_str_and_eval(self.user_params, self.user_params)
        list_module = []
        for dictwk in self.config["network"]:
            args   = dictwk.get("args")   if isinstance(dictwk.get("args"), list)   else ([dictwk.get("args")] if dictwk.get("args") is not None else [])
            kwargs = dictwk.get("kwargs") if isinstance(dictwk.get("kwargs"), dict) else {}
            args   = replace_sp_str_and_eval(args,   self.user_params)
            kwargs = replace_sp_str_and_eval(kwargs, self.user_params)
            if dictwk["class"] in __all__:
                kwargs["is_call_first"] = False
                dictwkwk = copy.deepcopy(self.user_params)
                for x in ["in_features", "__before"]:
                    if x in dictwkwk: del dictwkwk[x]
                if kwargs.get("user_parameters") is not None:
                    # If not None, partially overwrite
                    for x, y in kwargs["user_parameters"].items(): dictwkwk[x] = y
                kwargs["user_parameters"] = dictwkwk
                kwargs["dir_base_path"]   = self.dirpath
            print(dictwk["class"], args, kwargs)
            module = getattr(nn, dictwk["class"])(*args, **kwargs)
            if isinstance(module, nn.EvalModule): module.set_variables(self.shared_variables)
            ## Update out features 
            name_outnode = None
            if   dictwk.get("out_features") is None:
                name_outnode = "out_features"
            elif isinstance(dictwk.get("out_features"), int): self.user_params["__before"] = dictwk.get("out_features")
            elif isinstance(dictwk.get("out_features"), str):
                out_features = replace_sp_str_and_eval(dictwk.get("out_features"), self.user_params)
                if isinstance(out_features, int): self.user_params["__before"] = out_features
                else: name_outnode = out_features
            if name_outnode is not None:
                if hasattr(module, name_outnode): self.user_params["__before"] = getattr(module, name_outnode)
            list_module.append(module)
        setattr(self, self.name, nn.ModuleList(list_module))
        self.out_features      = self.user_params["__before"]
        self.is_debug          = False
        self.is_middle_release = False
        self.middle_mod        = []
        if self.is_call_first:
            for module in self.modules():
                if isinstance(module, nn.MiddleSaveOutput): self.middle_mod.append(module)
                if isinstance(module, nn.MiddleReleaseOutput):
                    if self.is_middle_release: raise("MiddleReleaseOutput can only be used once.")
                    self.is_middle_release = True
                if isinstance(module, nn.HuggingfaceModule):
                    self.tokenizer = module.tokenizer
                    self.huggingface_config = module.config
                if isinstance(module, nn.SharedParameterModule):
                    module.set_parameter({x:y for x, y in self.named_parameters()}[module.param_address])
    
    def forward(self, input: Union[torch.Tensor, List[torch.Tensor], dict]):
        if self.is_debug: t0 = time.perf_counter()
        output = input.clone() if self.is_call_first and isinstance(input, torch.Tensor) else input
        if self.is_middle_release:
            # We don't want to do the "isinstance" process multiple times, so we'll separate the process.
            for module in getattr(self, self.name):
                if isinstance(module, nn.MiddleReleaseOutput): output = self.middle_output()
                else: output = module(output)
        else:
            for module in getattr(self, self.name):
                output = module(output)
                if self.is_debug:
                    t1 = time.perf_counter()
                    print(f"mod: {module}\ntime: {t1 - t0}")
        if self.is_middle_release or len(self.middle_mod) == 0:
            return output
        else:
            return self.middle_output() + (output if isinstance(output, list) else [output])
    
    def middle_output(self):
        """
        Output the value saved from the layer defined by MiddleSaveOutput
        """
        output = [mod.middle_output for mod in self.middle_mod]
        for mod in self.middle_mod: mod.middle_output = None
        return output
    
    def debug(self, is_debug=True):
        """Don't use it."""
        for module in self.modules():
            if hasattr(module, "is_debug"):
                module.is_debug = is_debug
        return self

    def normal(self):
        """Don't use it."""
        return self.debug(is_debug=False)
    
    def forward_debug(self, input: torch.Tensor):
        """
        You can see the output of the layers defined in each
        """
        self.debug()
        t0 = time.perf_counter()
        try: self.forward(input)
        except Exception as e:
            print(type(e), e)
        list_output = []
        for module in self.modules():
            if isinstance(module, nn.BaseModule):
                list_output.append(module.save_output)
        t1 = time.perf_counter()
        print(f"forward_debug\ntime: {t1 - t0}")
        self.normal()
        return list_output

    def search_module(self, name: str):
        for module in self.modules():
            if isinstance(module, nn.BaseModule):
                if isinstance(module.name, str) and module.name == name:
                    return module
        return None


class RepeatModule(nn.Module):
    def __init__(self, *args, n_layers: int=1, **kwargs):
        super().__init__()
        assert isinstance(n_layers, int) and n_layers >= 1
        list_module = []
        for _ in range(n_layers):
            list_module.append(ConfigModule(*args, **kwargs))
            kwargs["in_features"] = list_module[-1].out_features
        self.list_module = nn.ModuleList(list_module)
    def forward(self, input: torch.Tensor):
        output = input
        for module in self.list_module: output = module(output)
        return output


class SplitModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.list_module = nn.ModuleList([ConfigModule(arg, **kwargs) for arg in args])
        self.is_check    = True
    def forward(self, input: List[torch.Tensor]):
        if self.is_check:
            # Check only the first time.
            assert isinstance(input, list) and len(input) == len(self.list_module)
            self.is_check = False
        list_output = []
        for i, module in enumerate(self.list_module):
            output = module(input[i])
            if isinstance(output, list):
                list_output += output
            else:
                list_output.append(output)
        return list_output


class ApplyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.apply_module = ConfigModule(*args, **kwargs)
        self.is_check     = True
    def forward(self, input: List[torch.Tensor]):
        if self.is_check:
            # Check only the first time.
            assert isinstance(input, list) or isinstance(input, tuple)
            self.is_check = False
        return [self.apply_module(x) for x in input]


class RegidualModule(nn.Module):
    def __init__(self, *args, **kwargs):
        assert len(args) == 2
        super().__init__()
        self.mod1 = ConfigModule(args[0], **kwargs)
        self.mod2 = ConfigModule(args[1], **kwargs)
    def forward(self, input: Union[torch.Tensor, List[torch.Tensor], dict]):
        return self.mod1(input) + self.mod2(input)