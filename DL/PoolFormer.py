import math
import numbers
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import wandb
import time
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from einops import rearrange, reduce, repeat
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
import timm.models
from timm.models.layers import DropPath,trunc_normal_


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    StarReLU from MetaFormer baseline for vision
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class MLPBlock(nn.Module):

  def __init__(self, d_model, mlp_ratio, drop_prob=0.1,act_layer=StarReLU):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_model*mlp_ratio)
    self.linear2 = nn.Linear(d_model*mlp_ratio, d_model)
    self.act_layer = act_layer()
    self.dropout1 = nn.Dropout(p=drop_prob)
    self.dropout2 = nn.Dropout(p=drop_prob)

  def forward(self, x):
    x = self.linear1(x)
    x = self.act_layer(x)
    x = self.dropout1(x)
    x = self.linear2(x)
    x = self.dropout2(x)
    return x

class MixedPooling(nn.Module):
    def __init__(self,pool_size,alpha=0.5,stride=1):
        super().__init__()
        self.pool_size=pool_size
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.padding=pool_size//2
        self.stride=stride

    def forward(self,x):
        x=self.alpha*F.max_pool2d(x,self.pool_size,self.stride,self.padding) + (1-self.alpha)*F.avg_pool2d(x,self.pool_size,self.stride,self.padding)
        return x

class MixedCustomPooling(nn.Module):
    def __init__(self,pool_size,alpha=0.5,stride=1):
        super().__init__()
        self.pool_size=pool_size
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.padding=pool_size//2
        self.stride=stride

    def forward(self,input):
        x = input.permute(0, 3, 1, 2)
        x=self.alpha*F.max_pool2d(x,self.pool_size,self.stride,self.padding) + (1-self.alpha)*F.avg_pool2d(x,self.pool_size,self.stride,self.padding)
        x = x.permute(0, 2, 3, 1)
        return x




class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, ybias=True,affine=True):
        super().__init__()
        if isinstance(normalized_shape,numbers.Integral):
            normalized_shape= (normalized_shape,)
        self.normalized_shape = tuple (normalized_shape)
        self.eps = eps
        self.affine = affine
        self.ybias = ybias
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            #self.bias = nn.Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            #self.register_parameter('bias', None)
        if self.ybias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
        if self.ybias:
            nn.init.zeros_(self.bias)

    def forward(self, input) :
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

class PatchEmbeddings(nn.Module):
    def __init__(self,patch_size=16, stride=16, padding=0, in_channels=3, embed_dim=768,
                 norm_layer=CustomLayerNorm,affine_norm_layer=True, ybias_norm_layer=True,
                 eps_norm_layer: float = 1e-5):
        super().__init__()
        self.proj=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=stride,padding=padding)
        self.norm = norm_layer(embed_dim,eps=eps_norm_layer,ybias=ybias_norm_layer,affine=affine_norm_layer) if norm_layer else nn.Identity()

    def forward(self,x):
        x=self.proj(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x=self.norm(x)
        return x

class Downsampling(nn.Module):

    def __init__(self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        affine_norm_layer=True, ybias_norm_layer=True, eps_norm_layer=1e-5,
        norm=None,act_layer=None,pre_permute=False):
        super().__init__()
        self.norm = norm(in_channels,eps=eps_norm_layer,ybias=ybias_norm_layer,affine=affine_norm_layer) if norm else nn.Identity()
        self.act_layer = act_layer() if act_layer else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)


    def forward(self, x):
        x=self.act_layer(x)
        x = self.norm(x)
        if self.pre_permute:
            #  [B, H, W, C] --> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]

        return x



class Scale(nn.Module):

    def __init__(self, dim, init_value=1e-5, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class MixedPoolFormerBlock(nn.Module):


    def __init__(self, dim,
                 token_mixer=MixedCustomPooling, mlp_ratio=4,
                 pool_size=3, alpha=0.5,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(pool_size=pool_size,alpha=alpha)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(d_model=dim,mlp_ratio=mlp_ratio,drop_prob=drop)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x

def make_mix_PoolFormer_blocks(dim, index, layers,
                 pool_size=3,alpha=0.5, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 res_scale_init_value=[1.0,1.0,1.0,1.0], layer_scale_init_value=1e-5):
    """
    Funzione che uso per creare diversi poolformer block con lunghezza di sequenza differente

    """
    blocks = []
    for _ in range(layers[index]):
        '''
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        '''
        blocks.append(MixedPoolFormerBlock(dim,pool_size=pool_size,alpha=alpha,mlp_ratio=mlp_ratio,
                                           norm_layer=norm_layer,drop=drop_rate,drop_path=drop_path_rate,
                                           layer_scale_init_value=layer_scale_init_value,res_scale_init_value=res_scale_init_value))

    blocks = nn.Sequential(*blocks)

    return blocks


