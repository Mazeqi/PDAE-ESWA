import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch
import torch.nn as nn
import torch.nn.functional as F

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        in_channels = 1000,
        flow_steps = 2,
        input_size = 256,
        conv3x3_only=False,
        hidden_ratio=1.0,
    ):
        super(FastFlow, self).__init__()
      
        self.nf_flows= nf_fast_flow(
                    [in_channels, int(input_size), int(input_size)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )

    def forward(self, features):
        output, log_jac_dets = self.nf_flows(features)

        loss = torch.mean(
            0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
        )

        log_prob = -(output**2) * 0.5
        prob = -torch.exp(log_prob)
            
        return prob
