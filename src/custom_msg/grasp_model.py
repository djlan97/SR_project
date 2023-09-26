#!usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
#from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_

import numbers
import math




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
    

class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        affine_norm_layer=True, ybias_norm_layer=True, eps_norm_layer=1e-5,
        norm=None,act_layer=None,pre_permute=False):
        super().__init__()
        self.norm = norm(in_channels,eps=eps_norm_layer,ybias=ybias_norm_layer,affine=affine_norm_layer) if norm else nn.Identity()
        self.act_layer = act_layer() if act_layer else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,
                                       stride=stride,padding=padding)



    def forward(self, x):
        x=self.act_layer(x)
        x = self.norm(x)
        if self.pre_permute:
            #  [B, H, W, C] --> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]

        return x

class PatchEmbeddingsUp(nn.Module):
    def __init__(self,patch_size=16, stride=16, padding=0, in_channels=3, embed_dim=768,
                 norm_layer=CustomLayerNorm,affine_norm_layer=True, ybias_norm_layer=True,
                 eps_norm_layer: float = 1e-5):
        super().__init__()
        self.proj=nn.ConvTranspose2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=stride,padding=padding)
        self.norm = norm_layer(embed_dim,eps=eps_norm_layer,ybias=ybias_norm_layer,affine=affine_norm_layer) if norm_layer else nn.Identity()

    def forward(self,x):
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        x=self.proj(x)


        return x

class AttentionBlockEncoder(nn.Module):
    def __init__(self,dim,n_head=None,head_dim=32,attn_drop=0.,out_drop=0.):
        super().__init__()
        self.n_head = n_head if n_head else dim // head_dim
        self.head_dim=head_dim
        self.attn_dim = self.n_head * self.head_dim
        #self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(dim,self.attn_dim)
        self.w_k = nn.Linear(dim,self.attn_dim)
        self.w_v = nn.Linear(dim,self.attn_dim)
        self.out = nn.Linear(self.attn_dim, dim)
        self.attn_drop=nn.Dropout(attn_drop)
        self.out_drop=nn.Dropout(out_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v):
        B, H, W, C = q.shape
        N = H * W
        q=self.w_q(q).reshape(B,N,self.n_head,self.head_dim).transpose(1,2)
        k = self.w_k(k).reshape(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(v).reshape(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k_t = k.transpose(2, 3)  # transpose
        score=torch.matmul(q,k_t)
        score = score / math.sqrt(self.head_dim)
        score = self.softmax(score)
        score = self.attn_drop(score)
        x = torch.matmul(score, v).transpose(1, 2).reshape(B, H, W, self.attn_dim)
        x = self.out(x)
        x=self.out_drop(x)
        return x,k,v

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
        #self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(d_model=dim,mlp_ratio=mlp_ratio,drop_prob=drop)

        #self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()
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

class AttentionMetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = AttentionBlockEncoder(dim=dim,attn_drop=drop,out_drop=drop)
        #self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(d_model=dim,mlp_ratio=mlp_ratio,drop_prob=drop)
        #self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self,x):
        y=self.norm1(x)
        y,k,v=self.token_mixer(q=y,k=y,v=y)
        x = self.res_scale1(x) + self.layer_scale1(self.drop_path1(y))

        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x,k,v

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

def make_AttentionMetaFormer_blocks(dim, index, layers,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 res_scale_init_value=1e-5, layer_scale_init_value=1e-5):
    """
    Funzione che uso per creare diversi poolformer block con lunghezza di sequenza differente

    """
    blocks = []
    for _ in range(layers[index]):
        '''
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        '''
        blocks.append(AttentionMetaFormerBlock(
            dim,mlp_ratio=mlp_ratio,norm_layer=norm_layer,
            drop=drop_rate,drop_path=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,res_scale_init_value=res_scale_init_value))

    blocks = nn.Sequential(*blocks)

    return blocks


class MPA_SegmentationNetwork(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None,
                 pool_size=3,
                 alpha=0.5,
                 in_channels=3,
                 res_scale_init_value=[1.0,1.0,1.0,1.0],
                 patch_size=4, stride=4, padding=0,
                 up_patch_size=2, up_stride=2, up_pad=0,
                 num_classes=1,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 layer_scale_init_value=None,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.patch_emb=PatchEmbeddings(patch_size=patch_size,stride=stride,padding=padding,in_channels=in_channels,
                                       embed_dim=embed_dims[0],ybias_norm_layer=False,eps_norm_layer=1e-6)
        self.block0=make_mix_PoolFormer_blocks(embed_dims[0],0,layers,pool_size=pool_size,alpha=alpha,mlp_ratio=mlp_ratios[0],drop_rate=drop_rate,
                                               drop_path_rate=drop_path_rate,res_scale_init_value=res_scale_init_value[0],
                                               layer_scale_init_value=layer_scale_init_value)

        self.dws0=Downsampling(kernel_size=down_patch_size,stride=down_stride,padding=down_pad,in_channels=embed_dims[0],out_channels=embed_dims[1],
                               pre_permute=True,norm=CustomLayerNorm,ybias_norm_layer=False,eps_norm_layer=1e-6)
        self.block1=make_mix_PoolFormer_blocks(embed_dims[1],1,layers,pool_size=pool_size,alpha=alpha,mlp_ratio=mlp_ratios[1],drop_rate=drop_rate,
                                               drop_path_rate=drop_path_rate,res_scale_init_value=res_scale_init_value[1],
                                               layer_scale_init_value=layer_scale_init_value)

        self.dws1 = Downsampling(kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                                 in_channels=embed_dims[1], out_channels=embed_dims[2],
                                 pre_permute=True, norm=CustomLayerNorm, ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block21 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[2])
        self.block22 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[2])
        self.block23 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[2])

        self.dws2 = Downsampling(kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                                 in_channels=embed_dims[2], out_channels=embed_dims[3],
                                 pre_permute=True, norm=CustomLayerNorm, ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block31 = make_AttentionMetaFormer_blocks(embed_dims[3], 3, layers, mlp_ratio=mlp_ratios[3],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[3])
        self.block32 = make_AttentionMetaFormer_blocks(embed_dims[3], 3, layers, mlp_ratio=mlp_ratios[3],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[3])
        self.block33 = make_AttentionMetaFormer_blocks(embed_dims[3], 3, layers, mlp_ratio=mlp_ratios[3],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[3])


        self.upsampling2 = Upsampling(in_channels=embed_dims[3], out_channels=embed_dims[2], kernel_size=up_patch_size,
                                      stride=up_stride, padding=up_pad, pre_permute=True, norm=CustomLayerNorm,
                                      ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.decoder_block21 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[2])
        self.decoder_block22 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                       drop_rate=drop_rate,
                                                       drop_path_rate=drop_path_rate,
                                                       layer_scale_init_value=layer_scale_init_value,
                                                       res_scale_init_value=res_scale_init_value[2])
        self.decoder_block23 = make_AttentionMetaFormer_blocks(embed_dims[2], 2, layers, mlp_ratio=mlp_ratios[2],
                                                               drop_rate=drop_rate,
                                                               drop_path_rate=drop_path_rate,
                                                               layer_scale_init_value=layer_scale_init_value,
                                                               res_scale_init_value=res_scale_init_value[2])

        self.upsampling1 = Upsampling(in_channels=embed_dims[2], out_channels=embed_dims[1], kernel_size=up_patch_size,
                                      stride=up_stride, padding=up_pad, pre_permute=True, norm=CustomLayerNorm,
                                      ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.decoder_block1 = make_mix_PoolFormer_blocks(embed_dims[1], 1, layers, pool_size=pool_size, alpha=alpha,
                                                         mlp_ratio=mlp_ratios[1], drop_rate=drop_rate,
                                                         drop_path_rate=drop_path_rate,
                                                         res_scale_init_value=res_scale_init_value[1],
                                                         layer_scale_init_value=layer_scale_init_value)
        self.upsampling0 = Upsampling(in_channels=embed_dims[1], out_channels=embed_dims[0], kernel_size=up_patch_size,
                                      stride=up_stride, padding=up_pad, pre_permute=True, norm=CustomLayerNorm,
                                      ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.decoder_block0 = make_mix_PoolFormer_blocks(embed_dims[0], 0, layers, pool_size=pool_size, alpha=alpha,
                                                         mlp_ratio=mlp_ratios[0], drop_rate=drop_rate,
                                                         drop_path_rate=drop_path_rate,
                                                         res_scale_init_value=res_scale_init_value[0],
                                                         layer_scale_init_value=layer_scale_init_value)
        self.patch_up = PatchEmbeddingsUp(patch_size=patch_size, stride=stride, padding=padding,
                                          in_channels=embed_dims[0],
                                          embed_dim=embed_dims[0], ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.skip0 = nn.Linear(in_features=embed_dims[0] * 2, out_features=embed_dims[0])
        self.skip1 = nn.Linear(in_features=embed_dims[1] * 2, out_features=embed_dims[1])
        self.skip2 = nn.Linear(in_features=embed_dims[2] * 2, out_features=embed_dims[2])
        # self.skip0 = nn.Conv2d(in_channels=embed_dims[0]*2,out_channels=embed_dims[0],kernel_size=1)
        # self.skip1 = nn.Conv2d(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1], kernel_size=1)
        # self.skip2 = nn.Conv2d(in_channels=embed_dims[2] * 2, out_channels=embed_dims[2], kernel_size=1)
        self.pos_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.cos_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.sin_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.width_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1,
                                      bias=False)

    def forward(self,x):
        x = self.patch_emb(x)
        x0 = self.block0(x)
        x = self.dws0(x0)
        x1 = self.block1(x)
        x = self.dws1(x1)
        x, _, _ = self.block21(x)
        x, _, _ = self.block22(x)
        x2,_,_ = self.block23(x)
        x = self.dws2(x2)
        x,_,_ = self.block31(x)
        x, _, _ = self.block32(x)
        x,_,_ = self.block33(x)
        x = self.upsampling2(x)
        x = torch.cat((x2, x), dim=-1)
        x = self.skip2(x)
        x, _, _ = self.decoder_block21(x)
        x,_,_ = self.decoder_block22(x)
        x,_,_ = self.decoder_block23(x)
        x = self.upsampling1(x)
        x = torch.cat((x1, x), dim=-1)
        x = self.skip1(x)
        x = self.decoder_block1(x)
        x = self.upsampling0(x)
        x = torch.cat((x0, x), dim=-1)
        x = self.skip0(x)
        x = self.decoder_block0(x)
        x = self.patch_up(x)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        # print("pos shape:",pos_pred.shape)
        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }



