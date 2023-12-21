# import clip
from .CLIP import clip
import torch
import torch.nn as nn
from .ResNet import *
from .resnet20_cifar import resnet20
import torch.nn.functional as F

from .CLIP.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

#copied from NC-FSCIL
class Projector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Projector, self).__init__()
        self.ln1 = nn.Sequential(
                       nn.Linear(in_channels, 2*in_channels, bias=False),
                       nn.LayerNorm( 2*in_channels),
                       nn.LeakyReLU(0.1)
        )
        self.ln2 = nn.Sequential(
                       nn.Linear(2 * in_channels, 2*in_channels, bias=False),
                       nn.LayerNorm( 2*in_channels),
                       nn.LeakyReLU(0.1)
        )
        self.ln3 = nn.Linear(in_channels * 2, out_channels, bias=False)
        self.ffn = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        identity = x
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = x + self.ffn(identity)
        return x


# from timm.models.layers import PatchEmbed

class MyModel(nn.Module):

    def __init__(self, 
                 dataset: str='cub_200', 
                 arch_name: str='ViT-B/32', 
                 prompt_len: int=8,
                 version: str='V1'):
        super(MyModel, self).__init__()
        self.version = version
        self.arch_name = arch_name
        # initi incremental info
        if dataset == 'miniImageNet' or dataset == 'cifar100':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 100, 60, 5, 9
            self.pretrained = False
        elif dataset == 'cub_200' or dataset == 'ImageNet_R':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 200, 100, 10, 11
            self.pretrained = False
        elif dataset == 'mnist' or dataset == 'cifar10':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 10, 8, 1, 3
            self.pretrained = False
        elif dataset == 'flowers':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 102, 62, 5, 9
            self.pretrained = False
        elif dataset == 'food101':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 101, 61, 5, 9
            self.pretrained = False
        elif dataset == 'car196':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 196, 156, 5, 9
            self.pretrained = True
        elif dataset == 'aircraft102':
            self.num_cls, self.base_cls_num, self.inc_cls_num, self.sessions = 100, 60, 5, 9
            self.pretrained = False
        else:
            raise Exception("Invalid dataset name {}".format(dataset))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # init model
        if arch_name == 'ResNet18':
            # import pdb
            # pdb.set_trace()
            self.backbone = resnet18(self.pretrained)
            self.num_features = 512
        elif arch_name == 'ResNet20':
            self.backbone = resnet20()
            self.num_features = 64
        else: # otherwise all the CLIP
            self.encoder, self.preprocess = clip.load(arch_name, device="cpu")
            self.hdim         = 768
            self.num_features = 512
        if 'ViT' not in self.arch_name:
            # self.dims = [64, 128, 256, 512]
            # if self.dims[int(cap_layer)] == self.num_features:
            #     self.proj = nn.Identity()
            # else:
            #     self.proj = nn.Linear(self.dims[int(cap_layer)], 512)

            # self.proj = nn.Linear(768, 512)

            # self.proj = nn.Linear(768, 512, bias=False)

            self.proj = Projector(768, 512)
            
        self.protos = nn.Linear(self.num_features, self.num_cls, bias=False)
        self.linear = nn.Linear(self.num_features, self.num_features, bias=False)

        # used to store specific knowledge
        self.heads_vis = nn.ModuleList()
        self.heads_vis.append(nn.Linear(self.num_features, self.base_cls_num, bias=False))
        for i in range(self.sessions-1):
            self.heads_vis.append(nn.Linear(self.num_features, self.inc_cls_num, bias=False))

        if 'ViT' in self.arch_name:
            # initialize visual prompts
            self.prompt = torch.randn((prompt_len, 768), requires_grad=True)
            self.prompt = nn.Parameter(self.prompt)
            self.prompt_len = prompt_len

    def encode_image(self, x: torch.Tensor, 
                    memory = None,
                    KPM = None,
                    cap_layer: int=-1,
                    upd_layer: int=-1,
                    upd_targt: str='none',
                    enable_prompt:bool=False,
                    return_all: bool=False,
                    linear: bool = False
                    ):
        if 'ViT' not in self.arch_name:
            x = self.backbone(x)
            x = self.avgpool(x).squeeze(-1).squeeze(-1)
            return x
        
        x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.visual.positional_embedding.to(x.dtype)

        if enable_prompt:
            # concat behind last token
            x = torch.cat((x, self.prompt.unsqueeze(0).repeat(x.shape[0],1 ,1)), dim=1) 
            # concat before cls token
            # x = torch.cat((self.prompt.unsqueeze(0).repeat(x.shape[0],1 ,1), x), dim=1)

        x = self.encoder.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if cap_layer != -1 and self.version=='V1':
            x = self.encoder.visual.transformer(x,
                                                cap_layer=cap_layer,
                                                prompt_len=self.prompt_len)
            return x
        else:
            x = self.encoder.visual.transformer(x,
                                                memory=memory, 
                                                KPM=KPM, 
                                                cap_layer=cap_layer,
                                                upd_layer=upd_layer,
                                                upd_targt=upd_targt,
                                                prompt_len=self.prompt_len,
                                                version=self.version)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.encoder.visual.ln_post(x) # modified

        if self.encoder.visual.proj is not None:
            x = x @ self.encoder.visual.proj
        
        if not return_all:
            x = x[:, 0, :]
            # x = x[:, :self.prompt_len, :].mean(dim=1)

        if linear:
            x = self.linear(x)
        return x


    def forward(self, x, cur_session=0):
        """
        x: input data
        """
        feat_vis = self.encode_image(x)
        outs_vis = []
        for i in range(cur_session+1):
            outs_vis.append(
                F.linear(
                    F.normalize(feat_vis, p=2, dim=-1), F.normalize(self.heads_vis[i].weight, p=2, dim=-1)
                    ))
        outs_vis = torch.cat(outs_vis, dim=1)
        return outs_vis
    


