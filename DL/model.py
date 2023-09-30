import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_
from PoolFormer import  make_mix_PoolFormer_blocks
import numbers


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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size [int]: window size

    Returns:
        windows: (B*num_windows, window_size,window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size,window_size, C)
    )
    return windows



def window_reverse(windows, window_size,H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,

        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # (self.relative_position_bias_table).shape= 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing="ij"))  # coords.shape = 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # coords_flatten.shape = 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # relative_coords.shape = 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # relative_coords.shape = Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # sub required to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # relative_position_index.shape = Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        '''
        Args:
            x.shape = (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        '''
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,k,v

class WindowCrossAttention(nn.Module):
    """

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,

        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # (self.relative_position_bias_table).shape= 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing="ij"))  # coords.shape = 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # coords_flatten.shape = 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # relative_coords.shape = 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # relative_coords.shape = Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # sub required to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # relative_position_index.shape = Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,p_k,p_v, mask=None):

        '''
        Args:
            x.shape = (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        '''
        B_, N, C = x.shape
        q = (
            self.qkv(x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k=p_k
        v=p_v


        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,k,v



class SwinTransformerBlock(nn.Module):

    '''
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    '''

    def __init__(self, dim,input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=StarReLU, norm_layer=nn.LayerNorm,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = MLPBlock(d_model=dim,mlp_ratio=mlp_ratio,drop_prob=drop,act_layer=act_layer)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        #self.fused_window_process = fused_window_process

    def forward(self, x):

        B, H, W, C = x.shape
        #assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        #x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows,k,v = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp) #B H W C

        # reverse cyclic shift
        if self.shift_size > 0:

            #shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        else:
            #shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        if pad_r > 0 or pad_b > 0 :
            x = x[:,:H,:W,:]
        #x = x.view(B, H , W, C)
        x = self.res_scale1(shortcut) + self.layer_scale1(self.drop_path(x))

        # FFN
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )


        return x,k,v


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.

        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 layer_scale_init_value=None, res_scale_init_value=None,

                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth


        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,input_resolution=input_resolution,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 layer_scale_init_value=layer_scale_init_value, res_scale_init_value=res_scale_init_value,
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for idx,blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x,k1,v1 = blk(x)
                k2=k1
                v2 = v1
            else:
                x,k2,v2=blk(x)
            #x = blk(x)


        if self.downsample is not None:
            x = self.downsample(x)
        return x,k1,v1,k2,v2

class MPASegmentationNetwork(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 input_resolution=[(28,28),(14,14),(7,7)],
                 mlp_ratios=None,num_heads=[4, 10, 16],
                 pool_size=3,
                 alpha=0.5,
                 in_channels=3,
                 res_scale_init_value=[1.0,1.0,1.0,1.0],
                 patch_size=4, stride=4, padding=0,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 up_patch_size=2,up_stride=2,up_pad=0,
                 num_classes=1, layer_scale_init_value=None,
                 drop_rate=0., drop_path_rate=0.):
        super(MPASegmentationNetwork, self).__init__()
        self.num_classes = num_classes
        self.patch_emb = PatchEmbeddings(patch_size=patch_size, stride=stride, padding=padding, in_channels=in_channels,
                                         embed_dim=embed_dims[0], ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block0 = make_mix_PoolFormer_blocks(embed_dims[0], 0, layers, pool_size=pool_size, alpha=alpha,
                                                 mlp_ratio=mlp_ratios[0], drop_rate=drop_rate,
                                                 drop_path_rate=drop_path_rate,
                                                 res_scale_init_value=res_scale_init_value[0],
                                                 layer_scale_init_value=layer_scale_init_value)

        self.dws0 = Downsampling(kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                                 in_channels=embed_dims[0], out_channels=embed_dims[1],
                                 pre_permute=True, norm=CustomLayerNorm, ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block1 =BasicLayer(dim=embed_dims[1],input_resolution=input_resolution[0],depth=2,
                                num_heads=num_heads[0],window_size=7,mlp_ratio=mlp_ratios[1],
                                drop=drop_rate,drop_path=drop_path_rate,
                                res_scale_init_value=res_scale_init_value[1],
                                layer_scale_init_value=layer_scale_init_value)


        self.dws1 = Downsampling(kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                                 in_channels=embed_dims[1], out_channels=embed_dims[2],
                                 pre_permute=True, norm=CustomLayerNorm, ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block2 =BasicLayer(dim=embed_dims[2],input_resolution=input_resolution[1],depth=2,
                                num_heads=num_heads[1],window_size=7,mlp_ratio=mlp_ratios[2],
                                drop=drop_rate,drop_path=drop_path_rate,
                                res_scale_init_value=res_scale_init_value[2],
                                layer_scale_init_value=layer_scale_init_value)

        self.dws2 = Downsampling(kernel_size=down_patch_size, stride=down_stride, padding=down_pad,
                                 in_channels=embed_dims[2], out_channels=embed_dims[3],
                                 pre_permute=True, norm=CustomLayerNorm, ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.block3 =BasicLayer(dim=embed_dims[3],input_resolution=input_resolution[2],depth=1,
                                num_heads=num_heads[2],window_size=7,mlp_ratio=mlp_ratios[3],
                                drop=drop_rate,drop_path=drop_path_rate,
                                res_scale_init_value=res_scale_init_value[3],
                                layer_scale_init_value=layer_scale_init_value)

        self.upsampling2 = Upsampling(in_channels=embed_dims[3],out_channels=embed_dims[2],kernel_size=up_patch_size,
                                      stride=up_stride,padding=up_pad,pre_permute=True,norm=CustomLayerNorm,
                                      ybias_norm_layer=False,eps_norm_layer=1e-6)
        self.decoder_block2 =BasicLayer(dim=embed_dims[2],input_resolution=input_resolution[1],depth=2,
                                num_heads=num_heads[1],window_size=7,mlp_ratio=mlp_ratios[2],
                                drop=drop_rate,drop_path=drop_path_rate,
                                res_scale_init_value=res_scale_init_value[2],
                                layer_scale_init_value=layer_scale_init_value)


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
        self.patch_up = PatchEmbeddingsUp(patch_size=patch_size, stride=stride, padding=padding, in_channels=embed_dims[0],
                                         embed_dim=embed_dims[0], ybias_norm_layer=False, eps_norm_layer=1e-6)
        self.skip0 = nn.Linear(in_features=embed_dims[0]*2,out_features=embed_dims[0])
        self.skip1 = nn.Linear(in_features=embed_dims[1] * 2, out_features=embed_dims[1])
        self.skip2 = nn.Linear(in_features=embed_dims[2] * 2, out_features=embed_dims[2])
        #self.skip0 = nn.Conv2d(in_channels=embed_dims[0]*2,out_channels=embed_dims[0],kernel_size=1)
        #self.skip1 = nn.Conv2d(in_channels=embed_dims[1] * 2, out_channels=embed_dims[1], kernel_size=1)
        #self.skip2 = nn.Conv2d(in_channels=embed_dims[2] * 2, out_channels=embed_dims[2], kernel_size=1)
        self.pos_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.cos_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.sin_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.width_output = nn.Conv2d(in_channels=embed_dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward(self,x):
        x = self.patch_emb(x)
        x0 = self.block0(x)
        x = self.dws0(x0)
        x1,_,_,_,_ = self.block1(x)
        x = self.dws1(x1)
        x2,_,_,_,_= self.block2(x)
        x = self.dws2(x2)
        x,_,_,_,_ = self.block3(x)
        x = self.upsampling2(x)
        x = torch.cat((x2,x),dim=-1)
        x = self.skip2(x)
        x,_,_,_,_ = self.decoder_block2(x)
        x = self.upsampling1(x)
        x = torch.cat((x1,x),dim=-1)
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


if __name__ == '__main__':

    inp = torch.randn((2,3,480,640))
    model = MPASegmentationNetwork(layers=[2,2,2,2],embed_dims=[64,128,320,512],
                                   mlp_ratios=(4,4,4,4),num_classes=1)
    pos_output, cos_output, sin_output, width_output = model(inp)
    print(pos_output.shape)
    print(cos_output.shape)
    print(sin_output.shape)
    print(width_output.shape)
    '''
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = (4, 4, 4, 4)
    input_resolution = [(28, 28), (14, 14), (7, 7)]
    num_heads = [32, 12, 24]
    layer_scale_init_value = None
    res_scale_init_value = [1.0,1.0,1.0,1.0]
    drop_rate = 0.
    drop_path_rate = 0.
    inp = torch.randn((1,28,28,128))
    model = BasicLayer(dim=embed_dims[1],input_resolution=input_resolution[0],depth=2,
                                num_heads=num_heads[0],window_size=7,mlp_ratio=mlp_ratios[1],
                                drop=drop_rate,drop_path=drop_path_rate,
                                res_scale_init_value=res_scale_init_value[1],
                                layer_scale_init_value=layer_scale_init_value)
    x2, _, _, _, _ = model(inp)
    print(x2.shape)
    '''




