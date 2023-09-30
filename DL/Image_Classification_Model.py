import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
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

class MLPBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, hidden_dim: int ,dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim,1)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_dim, in_dim,1)
        self.dropout_2 = nn.Dropout(dropout)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.act(x)
        x=self.dropout_1(x)
        x=self.conv2(x)
        x=self.dropout_2(x)
        return x

#Definizione MixedPooling
class MixedPooling(nn.Module):
    def __init__(self,pool_size,alpha=0.5,stride=1):
        super().__init__()
        self.kernel_size=pool_size
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.padding=pool_size//2
        self.stride=stride

    def forward(self,x):
        x=self.alpha*F.max_pool2d(x,self.kernel_size,self.stride,self.padding) + (1-self.alpha)*F.avg_pool2d(x,self.kernel_size,self.stride,self.padding)
        return x

# Personalizzazione del GroupNorm con un solo gruppo
class CustomGroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

#Definizione PoolFormerBlock
class PoolFormerBlock(nn.Module):

    def __init__(self,dim,mlp_ratio=4,pool_size=3,alpha=0.5,norm_layer=CustomGroupNorm,dropout=0., drop_path=0.,use_layer_scale=True,layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = MixedPooling(pool_size=pool_size,alpha=alpha)
        self.norm2 = norm_layer(dim)
        mlp_dim=dim*mlp_ratio
        self.mlp=MLPBlock(in_dim=dim,hidden_dim=mlp_dim,dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Definizione del PE
class PatchEmbeddings(nn.Module):
    def __init__(self,patch_size=16, stride=16, padding=0, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=stride,padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        x=self.proj(x)
        x=self.norm(x)
        return x


def make_pool_blocks(dim, index, layers,
                 pool_size=3,alpha=0.5, mlp_ratio=4.,
                 norm_layer=CustomGroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    Funzione che uso per creare diversi poolformer block con lunghezza di sequenza differente

    """
    blocks = []
    for _ in range(layers[index]):
        '''
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        '''
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size,alpha=alpha, mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            dropout=drop_rate, drop_path=drop_path_rate,
            use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value))
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 pool_size=3,
                 alpha=0.5,
                 in_channels=3,
                 norm_layer=CustomGroupNorm,
                 num_classes=10,
                 patch_size=16, stride=16, padding=0,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.patch_emb=PatchEmbeddings(patch_size=patch_size,stride=stride,padding=padding,in_channels=in_channels,embed_dim=embed_dims[0])

        blocks=[]
        for i in range(len(layers)):
            stage=make_pool_blocks(embed_dims[i],i,layers,pool_size=pool_size,alpha=alpha,mlp_ratio=mlp_ratios[i],norm_layer=norm_layer,
                                   use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
                                   drop_rate=drop_rate,drop_path_rate=drop_path_rate)
            blocks.append(stage)

            if i>=len(layers)-1:
                break

            if downsamples[i] or embed_dims[i]!=embed_dims[i+1]:
                blocks.append(
                    PatchEmbeddings(patch_size=down_patch_size,stride=down_stride,padding=down_pad,in_channels=embed_dims[i],embed_dim=embed_dims[i+1])
                )
        self.blocks=nn.ModuleList(blocks)
        self.norm=norm_layer(embed_dims[-1])

        self.head=nn.Linear(embed_dims[-1],num_classes)

    def forward(self,x):
        x=self.patch_emb(x)
        for block in self.blocks:
            x=block(x)
        x=self.norm(x)

        x=self.head(x.mean([-2, -1]))
        return x

#Funzioni per creare le il PoolFormer con diverse quantità di layer
def MixPoolFormer_4(alpha=0.5):
    layer=[1, 1, 1, 1]
    emb_dim=[64,128,256,512]
    mlp_ratios = [2, 2, 2, 2]
    downsamples = [True, True, True, True]
    model = PoolFormer(
        layers=layer, embed_dims=emb_dim,alpha=alpha,
        mlp_ratios=mlp_ratios, downsamples=downsamples,num_classes=4)
    return model


from torchvision import datasets, models, transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        #transforms.CenterCrop(224),
        #transforms.RandomRotation(degrees=50),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset/dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in [ 'train','val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

if __name__ == "__main__":

    model = MixPoolFormer_4()
    #model = torch.load('Weights/model_classification.pth')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=35, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=30)
    torch.save(model,'Weights/model_classification.pth')

    #print(class_names)

