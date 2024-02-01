from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SeqConvNorm, SeqDeconv2d
from .convlstm import ConvLSTM
        
        
class VidoePredictorAbs(ABC, nn.Module):
    KERNEL_SIZE=5
    RELU_SHIFT=1e-12
    def __init__(self, img_resolution=64, use_state=True, 
                 num_masks=10, color_channels=3, state_dim=5, action_dim=5):
        super(VidoePredictorAbs, self).__init__()
        
        self.img_res = img_resolution
        self.use_state = use_state
        self.num_masks = num_masks
        
        self.conv1 = SeqConvNorm(3, 32, self.KERNEL_SIZE, stride=2, padding=2, 
                                 out_dim=img_resolution//2) # 32x32
        self.lstm1 = ConvLSTM(32, 32, self.KERNEL_SIZE, num_layers=1, 
                              out_dim=img_resolution//2) # 32x32
        self.lstm2 = ConvLSTM(32, 32, self.KERNEL_SIZE, num_layers=1, 
                               out_dim=img_resolution//2) # 32x32
        self.conv2 = SeqConvNorm(32, 64, 3, stride=2, padding=1, 
                                 out_dim=img_resolution//4, norm=False)  # 16x16
        self.lstm34 = ConvLSTM(64, 64, self.KERNEL_SIZE, num_layers=2,
                               out_dim=img_resolution//4) # 16x16
        self.conv3 = SeqConvNorm(64, 128, 3, stride=2, padding=1,
                                 out_dim=img_resolution//8, norm=False) # 8x8
        channel = 128 if not use_state else 128+10
        self.conv4 = SeqConvNorm(channel, channel, 1, stride=1, padding=0, 
                                 out_dim=img_resolution//8, norm=False) # 8x8
        self.lstm5 = ConvLSTM(channel, channel, self.KERNEL_SIZE, num_layers=1,
                              out_dim=img_resolution//8) # 8x8
        self.deconv1 = SeqDeconv2d(channel, 64, out_dim=img_resolution//4,
                                   stride=2, kernel_size=3, norm=False)
        self.lstm6 = ConvLSTM(64, 64, self.KERNEL_SIZE, num_layers=1, 
                              out_dim=img_resolution//4) # 16x16
        self.deconv2 = SeqDeconv2d(64*2, 32, out_dim=img_resolution//2,
                                   stride=2, kernel_size=3, norm=False) # 32x32
        self.lstm7 = ConvLSTM(32, 32, self.KERNEL_SIZE, num_layers=1, 
                              out_dim=img_resolution//2) # 32x32
        self.deconv3 = SeqDeconv2d(32*2, 32, out_dim=img_resolution,
                                   kernel_size=3, stride=2, norm=True) # 64x64
        
        # mask prediction
        self.deconv4 = nn.ConvTranspose2d(32, num_masks+1, 1)
        
        
        self.deconv5 = nn.ConvTranspose2d(32, color_channels, 1)
        
        # motion prediction
        self.fcm = nn.Linear(channel*((img_resolution//8)**2), (self.KERNEL_SIZE**2)*num_masks)
        
        # State prediction
        self.state_fc = nn.Linear(state_dim+action_dim, state_dim)
        
    def forward(self, x, state=None, action=None):
        # x [B,SEQ,C,H,W]
        # state [B,SEQ,STATE_DIM=5]
        # action [B,SEQ,ACTION_DIM=5]
        if state is None:
            state = torch.zeros((x.size(0), x.size(1), 5), 
                                dtype=x.dtype, device=x.device)
        if action is None:
            action = torch.zeros((x.size(0), x.size(1), 5), 
                                 dtype=x.dtype, device=x.device)

        state_action = torch.concat([state, action], dim=-1).view(x.size(0), x.size(1),
                                                 state.size(-1)+action.size(-1), 
                                                 1, 1)
        skip1 = self.conv1(x)
        t = self.lstm1(skip1)[1][0]
        skip2 = self.conv2(self.lstm2(t)[1][0])
        t = self.conv3(self.lstm34(skip2)[1][0])
        if self.use_state:
            sa = torch.tile(state_action, (1, 1, 1, t.size(-2), t.size(-1)))
            t = torch.concat([t, sa], dim=2)
        tt = self.lstm5(self.conv4(t))[1][0]    
        
        t = self.lstm6(self.deconv1(tt))[1][0]
        t = self.deconv2(torch.concat([t, skip2], dim=2))
        t = self.lstm7(t)[1][0]
        t = self.deconv3(torch.concat([t, skip1], dim=2))
        
        # motion prediction
        J_t, _ = self.transformation(x[:, -1], tt[:, -1], self.deconv5(t[:, -1]))
        
        if J_t.dim() == 4:
            J_t = J_t.unsqueeze(1)
        
        masks = self.deconv4(t[:,-1]).view(x.size(0), self.num_masks+1, -1)
        masks = torch.softmax(masks, dim=-1).view(x.size(0), self.num_masks+1, x.size(-2), x.size(-1))

        gen_image = x[:, -1] * masks[:,-1].unsqueeze(1).expand_as(x[:, -1])
        for i in range(self.num_masks):
            gen_image += J_t[:,i] * masks[:, i].unsqueeze(1).expand_as(J_t[:,i])
        
        return gen_image, self.state_fc(state_action.squeeze())
        
    @abstractmethod
    def transformation(self):
        pass
    
class CDNA(VidoePredictorAbs):
    def __init__(self, img_resolution=64, use_state=True, num_masks=10, color_channels=3,
                 state_dim=5, action_dim=5):
        super(CDNA, self).__init__(img_resolution, use_state, num_masks, 
                                   color_channels, state_dim, action_dim)
    
    def transformation(self, prev_img, latent, l_hidden):
        # latent [B, C, H, W]
        # prev_img [B, C, H, W]
        B, C, H, W = prev_img.shape
        transformed = torch.sigmoid(l_hidden).unsqueeze(1)
        num_masks = self.num_masks-1
        m = self.fcm(latent.view(B, -1)).view(B, num_masks, self.KERNEL_SIZE,
                                                self.KERNEL_SIZE)
        m = F.relu(m - self.RELU_SHIFT) + self.RELU_SHIFT
        m = m / torch.sum(m, dim=[2, 3], keepdim=True)
        
        prev_img = prev_img.permute([1, 0, 2, 3])
        m = m.view(B*num_masks, 1, self.KERNEL_SIZE, self.KERNEL_SIZE)
        j_t = F.conv2d(prev_img, m, padding='same', groups=B).reshape(C, B, num_masks, H, W)
        j_t = j_t.permute([1, 2, 0, 3, 4])
        
        transformed = torch.cat([transformed, j_t], dim=1)
        
        return j_t, m.view(B, num_masks, self.KERNEL_SIZE, self.KERNEL_SIZE)
    
class STP(VidoePredictorAbs):
    def __init__(self, img_resolution=64, use_state=True, num_masks=10, color_channels=3,
                 state_dim=5, action_dim=5):
        super(STP, self).__init__(img_resolution, use_state, num_masks, 
                                   color_channels, state_dim, action_dim)
        
        channel = 128 if not use_state else 128+10
        self.fcm = nn.Linear(channel*((img_resolution//8)**2), 100)
        self.stp_affine = nn.Linear(100, (self.num_masks-1)*6)
    
    def transformation(self, prev_img, latent, l_hidden):
        # latent [B, C, H, W]
        # prev_img [B, C, H, W]
        B, C, H, W = prev_img.shape
        transformed = torch.sigmoid(l_hidden).unsqueeze(1)
        
        m = self.fcm(latent.view(B, -1))
        m = self.stp_affine(m).view(B, self.num_masks-1, 2, 3)
        m += torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=m.dtype, device=m.device)
        j_t = []
        for i in range(self.num_masks-1):
            grid = F.affine_grid(m[:, i], prev_img.size())
            j_t.append(F.grid_sample(prev_img, grid, mode='bilinear', 
                                padding_mode='zeros', align_corners=False))

        j_t = torch.stack(j_t, dim=1)
        transformed = torch.cat([transformed, j_t], dim=1)

        return transformed, j_t
    
class DNA(VidoePredictorAbs):
    def __init__(self, img_resolution=64, use_state=True, num_masks=1, 
                 color_channels=3, state_dim=5, action_dim=5):
        super(DNA, self).__init__(img_resolution, use_state, num_masks, 
                         color_channels, state_dim, action_dim)
        
        assert self.num_masks == 1, "DNA only supports num_masks=1"
        
        self.deconv5 = nn.ConvTranspose2d(32, self.KERNEL_SIZE**2, 1)
        
    def transformation(self, prev_img, latent, l_hidden):
        # prev_img [B, C, H, W]
        # l_hidden [B, KERNEL_SIZE**2, H, W]
        B, C, H, W = prev_img.shape
        prev_img_pad = F.pad(prev_img, (2, 2, 2, 2))
        
        inputs = []
        for xkern in range(self.KERNEL_SIZE):
            for ykern in range(self.KERNEL_SIZE):
                # Slice the padded image
                inputs.append(prev_img_pad[:, :, xkern:xkern + H, ykern:ykern + W].unsqueeze(2))
        inputs = torch.cat(inputs, dim=2)  # [N, C, DNA_KERN_SIZE**2, H, W]

        # Normalize the kernels
        kernel = F.relu(l_hidden - self.RELU_SHIFT) + self.RELU_SHIFT
        kernel = kernel / torch.sum(kernel, dim=1, keepdim=True)
        kernel = kernel.unsqueeze(1)  # Add an extra dimension for broadcasting
        
        # Apply the kernels to the inputs and sum over the kernel dimension
        transformed = torch.sum(kernel * inputs, dim=2)  # [N, C, H, W]
        
        return transformed, kernel.squeeze()
        
        