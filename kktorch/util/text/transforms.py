import copy
from typing import Union, List
import torch

from kktorch.util.com import CheckFunction, check_type_list


__all__ = [
    "shift_right",
    "decoder_attention_mask",
    "generate"
]


class shift_right_decoder_input(CheckFunction):
    def check(self, input_ids: Union[List, torch.Tensor], decoder_start_token_id: int=0, eos_token_id: int=1, padding_token_id: int=0):
        assert isinstance(input_ids, torch.Tensor) or check_type_list(input_ids, list, int)
        assert isinstance(decoder_start_token_id, int)
        assert isinstance(eos_token_id, int)
        assert isinstance(padding_token_id, int)
        self.convert = lambda x: x
        if isinstance(input_ids, List):
            self.convert = lambda x: torch.Tensor(x)
    def main(self, input_ids: Union[List, torch.Tensor], decoder_start_token_id: int=0, eos_token_id: int=1, padding_token_id: int=0):
        # see: https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L770-L794
        input_ids = self.convert(input_ids)
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (eos_token_id,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        shifted_input_ids[shifted_input_ids == eos_token_id] = padding_token_id
        return shifted_input_ids


class decoder_attention_mask(CheckFunction):
    def check(self, input: Union[List, torch.Tensor], ignore_index: int):
        assert isinstance(input, torch.Tensor) or check_type_list(input, list, int)
        assert isinstance(ignore_index, int)
        self.convert = lambda x: x
        if isinstance(input, List):
            self.convert = lambda x: torch.Tensor(x)
    def main(self, input: Union[List, torch.Tensor], ignore_index: int):
        input  = self.convert(input)
        output = torch.tril(
            torch.ones(
                input.shape[0], input.shape[1], input.shape[1]
            )
        )
        output[input == ignore_index] = 0.0
        return output


class generate(CheckFunction):
    def check(self, model: torch.nn.Module, input: dict, bos_token_id: int=0, eos_token_id: int=1):
        assert isinstance(model, torch.nn.Module)
        assert isinstance(input, dict)
        assert isinstance(bos_token_id, int)
        assert isinstance(eos_token_id, int)
    def main(self, model: torch.nn.Module, input: dict, bos_token_id: int=0, eos_token_id: int=1, max_loop: int=256):
        device = next(model.parameters()).device
        input  = {x:y.to(device) if hasattr(y, "to") else y for x, y in copy.deepcopy(input).items()}
        if input.get("decoder_input_ids")      is not None: del input["decoder_input_ids"]
        if input.get("decoder_attention_mask") is not None: del input["decoder_attention_mask"]
        decoder_input_ids = torch.full((input["input_ids"].shape[0], 1), bos_token_id).to(device)
        input["decoder_input_ids"] = decoder_input_ids
        for _ in range(max_loop):
            output = model(input)
            output = torch.argmax(output, dim=2)[:, -1:]
            decoder_input_ids = torch.cat([decoder_input_ids, output], dim=1)
            if ((decoder_input_ids == eos_token_id).sum(dim=1) > 0).sum() == output.shape[0]:
                break
            input["decoder_input_ids"] = decoder_input_ids
        return decoder_input_ids
