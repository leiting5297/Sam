# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn.functional as F
import torch
import torch.nn as nn

from typing import Type

from einops import rearrange, repeat

class Adapter(nn.Module):
    
#     def __init__(self, input_size, hidden_size, output_size, num_attention_heads=4):
#         super(Adapter, self).__init__()
        
#         # Multi-head self-attention layer
#         self.multihead_attention = nn.MultiheadAttention(input_size, num_attention_heads)
        
#         # Feedforward neural network
#         self.feedforward = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
    
#     def forward(self, x):
#         # Multi-head self-attention
#         attn_output, _ = self.multihead_attention(x, x, x)
        
#         # Residual connection and layer normalization
#         x = x + attn_output
#         x = nn.LayerNorm(x.size()[1:])(x)
        
#         # Feedforward network
#         ff_output = self.feedforward(x)
        
#         # Residual connection and layer normalization
#         x = x + ff_output
#         x = nn.LayerNorm(x.size()[1:])(x)
        
#         return x
    
    
    
    def __init__(self, D_features,h,w, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        # Multi-head self-attention layer
        self.multihead_attention = nn.MultiheadAttention(768, 4)
        
        self.norm1 = nn.LayerNorm(768)

        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 768)

        # Layer normalization for the MLP output
        self.norm2 = nn.LayerNorm(768)
        
        
   
        
        
#         D_hidden_features = int(D_features * mlp_ratio)
#         self.act = act_layer()
#         self.D_fc1 = nn.Linear(D_features, D_hidden_features)
#         self.D_fc2 = nn.Linear(D_hidden_features, D_features)
#         self.FF=FFParser(D_features,h,int(w/2+1))
    def forward(self, x):
        xs=x
        x = self.norm1(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        x = xs + x

            # 调整形状回原始维度
        return x


class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, 3, 1024, 513, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        
     
        
        x= rearrange(x, 'b h w c -> b c w h').contiguous()
        
        B, C, H, W = x.shape
        
#         torch.Size([50, 14, 14, 768])
        
#         print(x.shape)
#       assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        
#         print("x_afterfft",x.size())
#         print("weight",weight.size())
        
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        
        
#         print("x_afterinv_fft",x.size())

        x= rearrange(x, 'b c w h -> b h w c').contiguous()

        return x
    
class Adapter_low(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        # xs = fft_low(x)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = fft_low(xs)
        xs = self.D_fc2(xs)
        # xs = fft_low(xs)
        # 这两个地方都可以，可以挑
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

def fft_high(x, rate=0.25):
    #print('fft测试',x.shape)
    #实际维度 batch,h,w,深度
    #要求输入维度：batch,深度，w,h
    x_base=x
    x_base= rearrange(x_base, 'b h w c -> b c w h').contiguous()
    # the smaller rate, the smoother; the larger rate, the darker
    # rate = 4, 8, 16, 32
    mask = torch.zeros(x_base.shape).to(x_base.device)
    w, h = x_base.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)
    mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x_base, norm="forward"))
    # mask[fft.float() > self.freq_nums] = 1
    # high pass: 1-mask, low pass: mask
    fft = fft * (1 - mask)
    # fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real

    inv = torch.abs(inv)
    inv=rearrange(inv, 'b c w h -> b h w c').contiguous()
    return inv

# def FF(x, spatial_size=None):
#     B, C, H, W = x.shape
#     complex_weight = nn.Parameter(torch.randn((2, 3), H, W, 2, dtype=torch.float32) * 0.02)
#     assert H == W, "height and width are not equal"
#     if spatial_size is None:
#         a = b = H
#     else:
#         a, b = spatial_size
#     # x = x.view(B, a, b, C)
#     x = x.to(torch.float32)
#     x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
#     weight = torch.view_as_complex(self.complex_weight)
#     x = x * weight
#     x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
#     x = x.reshape(B, C, H, W)
#     return x







def fft_low(x, rate=0.25):

    #实际维度 batch,h,w,深度
    #print('fft测试', x.shape)
    #要求输入维度：batch,深度，w,h
    x_base=x

    x_base= rearrange(x_base, 'b h w c -> b c w h').contiguous()


    # the smaller rate, the smoother; the larger rate, the darker
    # rate = 4, 8, 16, 32
    mask = torch.zeros(x_base.shape).to(x_base.device)
    w, h = x_base.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)
    mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

    fft = torch.fft.fftshift(torch.fft.fft2(x_base, norm="forward"))
    # mask[fft.float() > self.freq_nums] = 1
    # high pass: 1-mask, low pass: mask
    fft = fft * mask
    # fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    inv = torch.fft.ifft2(fft_hires, norm="forward").real

    inv = torch.abs(inv)
    inv=rearrange(inv, 'b c w h -> b h w c').contiguous()
    return inv

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


