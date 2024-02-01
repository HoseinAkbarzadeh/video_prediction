from collections.abc import Iterable
from itertools import repeat
import math

from torch.nn import Module, ModuleList, init, LayerNorm, Identity
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter
import torch

def _pair(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))
    
class ConvLSTMCell(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding

        self.input_weight = Parameter(torch.empty(4*out_channels, in_channels, *self.kernel_size))
        self.hidden_weight = Parameter(torch.empty(4*out_channels, out_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(4*out_channels))
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.input_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.hidden_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.input_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, hstate, cstate):
        # x shape: [B, C, H, W]
        x_t = conv2d(x, self.input_weight, self.bias, stride=self.stride, padding=self.padding)
        h_t = conv2d(hstate, self.hidden_weight, stride=self.stride, padding=self.padding)
        
        i = torch.sigmoid(x_t[:, :self.out_channels] + h_t[:, :self.out_channels])
        f = torch.sigmoid(x_t[:, self.out_channels:2*self.out_channels] + \
                            h_t[:, self.out_channels:2*self.out_channels])
        c = f * cstate + i * torch.tanh(x_t[:, 2*self.out_channels:-self.out_channels] + \
                                        h_t[:, 2*self.out_channels:-self.out_channels])
        o = torch.sigmoid(x_t[:, -self.out_channels:] + h_t[:, -self.out_channels:])
        h = o * torch.tanh(c)
        return h, [h, c]
    

class ConvLSTM(Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, 
                 batch_first=True, bias=True, out_dim=None) -> None:
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(ConvLSTMCell(in_ch, out_channels, kernel_size, 1, 'same', bias))
        
        self.layers = ModuleList(layers)
        
        if out_dim is None:
            self.norm = Identity()
        else:
            self.norm = LayerNorm([out_channels, *_pair(out_dim)])
        
    def forward(self, x):
        if x.dim() not in (4, 5):
            raise ValueError(f"Expected input to be 4D or 5D(batched) tensor, but got {x.dim()}D.")
        if x.dim() == 4:
            x = x.unsqueeze()
        
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        for i in range(self.num_layers):
            hidden_state = torch.zeros((x.size(0), self.out_channels, x.size(3), x.size(4)), 
                                   device=x.device, dtype=x.dtype)
            cell_state = torch.zeros_like(hidden_state)
            
            hstates = []
            cstates = []
            
            for m in range(x.size(1)):
                _, (hidden_state, cell_state) = self.layers[i](x[:, m], hidden_state, cell_state)
                hstates.append(hidden_state)
                cstates.append(cell_state)
            
            hstates = torch.stack(hstates, dim=1)
            cstates = torch.stack(cstates, dim=1)
            x = self.norm(hstates)
        return x[:, -1], (x, cstates)