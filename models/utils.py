import torch.nn as nn

from models.convlstm import _pair, ConvLSTM

class SeqConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int, padding: int, 
                 out_dim: int, norm: bool = True, **kwargs):
        super(SeqConvNorm, self).__init__()
        
        out_dim = _pair(out_dim)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, **kwargs)
        if norm:
            self.lnorm = nn.LayerNorm([out_channels, *out_dim])
        else:
            self.lnorm = nn.Identity()

        self.out_dim = out_dim
        self.out_channels = out_channels

    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B*L, C, H, W)
        x = self.conv(x)
        x = self.lnorm(x)
        x = x.view(B, L, self.out_channels, *self.out_dim)
        return x
    
class ConvLSTMNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_dim, 
                 num_layers=1, batch_first=True, bias=True) -> None:
        super(ConvLSTMNorm, self).__init__()
        
        out_dim = _pair(out_dim)
        
        self.convlstm = ConvLSTM(in_channels, out_channels, kernel_size, 
                             num_layers, batch_first, bias)
        
        self.lnorm = nn.LayerNorm([out_channels, *out_dim])

        self.out_dim = out_dim
        self.out_channels = out_channels
    
    def forward(self, x):
        _, (x, _) = self.convlstm(x)
        x = self.lnorm(x)
        return x
    
class SeqDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, out_dim, 
                 kernel_size=3, stride=2, 
                 padding=1, output_padding=1, norm=True):
        super(SeqDeconv2d, self).__init__()
        
        out_dim = _pair(out_dim)
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                         stride=stride, padding=padding, 
                                         output_padding=output_padding)
        if norm:
            self.lnorm = nn.LayerNorm([out_channels, *out_dim])
        else:
            self.lnorm = nn.Identity()
        
        self.out_dim = out_dim
        self.out_channels = out_channels
        
    def forward(self, x):
        B, L, C, H, W = x.shape
        x = x.view(B*L, C, H, W)
        x = self.deconv(x)
        x = self.lnorm(x)
        x = x.view(B, L, self.out_channels, *self.out_dim)
        return x