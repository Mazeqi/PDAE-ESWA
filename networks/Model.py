from torch.nn import Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, Tanh
from torch import nn
import yaml
from networks.common import *

    
class Model(nn.Module):
    def __init__(self, net_yaml_path='yolor-csp-c.yaml', in_ch = 3, final_ch = 0, out_layers = []): 
        super(Model, self).__init__()

        """Builds the neural network."""
        with open(net_yaml_path) as f:
            yaml_file = yaml.load(f, Loader=yaml.SafeLoader)
        network_list = yaml_file["network"]
        
        self.model, self.save = build_network(network_list, in_ch, final_ch)  # model, savelist
        self.out_layers = out_layers

    def forward(self, x):
        return self.forward_once(x) 

    def forward_once(self, x):
        y = []  # outputs
        out_x_layers = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            #print(x.shape)
            y.append(x if m.i in self.save else None)  # save output

            if len(self.out_layers) != 0:
                if m.i in self.out_layers:
                    out_x_layers.append(x)
            
        if len(self.out_layers) != 0:
            return x, out_x_layers
        else:
            return x

def build_network(network_list, in_ch = 3, final_ch = 0):
    

    layers, save_fea =  [], []  # layers, ch in, save the feature of some layers
    ch_list = [in_ch]
    out_ch = ch_list[-1]

    for i, (f, n, m, args) in enumerate(network_list):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        if m in [Conv2d, ConvTranspose2d, ConvTranspose2d_2, TConv2, Conv, TConv]:
            in_ch, out_ch = ch_list[f], args[0]
            if out_ch == 0:
                out_ch = final_ch
            args = [in_ch, out_ch, *args[1:]]
        elif m in [SumConvTranspose2d]:
            in_ch, out_ch = ch_list[-1], args[0]
            args = [in_ch, out_ch, *args[1:]]
        elif m is BatchNorm2d:
            args = [ch_list[f]]
        elif m in [SumBl]:
            out_ch = ch_list[f[0]] 
        else:
            out_ch = ch_list[f]

        # module
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args) 
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers.append(m_)
        if i == 0:
            ch_list = []
        ch_list.append(out_ch)
        save_fea.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    return nn.Sequential(*layers), sorted(save_fea)
