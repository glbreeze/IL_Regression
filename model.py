import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F


class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2, args=None):
        super(RegressionResNet, self).__init__()
        self.args = args 

        if args.arch == 'resnet18' or args.arch == 'res18':
            resnet_model = models.resnet18(pretrained=pretrained)
        elif args.arch == 'resnet50' or args.arch == 'res50':
            resnet_model = models.resnet50(pretrained=pretrained)
        
        if args.dataset in ['mnist']:
            conv1_out_ch = resnet_model.conv1.out_channels
            resnet_model.conv1 = nn.Conv2d(1, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)  
            resnet_model.maxpool = nn.Identity()
            
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])
        
        if self.args.feat == 'f':    # without GAP 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
                nn.Flatten(), 
                nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features)
            )
        else: 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
                nn.Flatten()
            )

        self.fc = nn.Linear(resnet_model.fc.in_features, num_outputs, bias=args.bias)
        if self.args.init_s > 0 and self.args.init_s != 1:
            self.fc.weight.data = self.fc.weight.data * self.args.init_s
    
    def forward(self, x, ret_feat=False):
        x = self.backbone(x)
        feat = self.feat(x)
        out = self.fc(feat)
        if ret_feat:
            return out, feat
        else:
            return out

    def forward_feat(self, x):
        out1 = self.backbone[:4](x)
        out2 = self.backbone[4](out1)
        out3 = self.backbone[5](out2)
        out4 = self.backbone[6](out3)
        out5 = self.backbone[7](out4)
        return [F.adaptive_avg_pool2d(out5, (1, 1)).view(out5.size(0), -1),
                F.adaptive_avg_pool2d(out4, (1, 1)).view(out4.size(0), -1),
                F.adaptive_avg_pool2d(out3, (1, 1)).view(out3.size(0), -1),
                F.adaptive_avg_pool2d(out2, (1, 1)).view(out2.size(0), -1)
        ]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, args, arch='256_256', max_action=1.0):
        super(MLP, self).__init__()
        # in_dim is state_dim
        self.args = args
        module_list = []
        for i, hidden_size in enumerate(arch.split('_')):
            hidden_size = int(hidden_size)
            module_list.append(nn.Linear(in_dim, hidden_size))
            if args.bn == 't':
                module_list.append(nn.BatchNorm1d(hidden_size, affine=False))
            elif args.bn == 'p': 
                module_list.append(nn.BatchNorm1d(hidden_size, affine=True))
            if args.act == 'elu': 
                module_list.append(nn.ELU())
            elif args.act == 'lrelu': 
                module_list.append(nn.LeakyReLU(0.1))
            else: 
                module_list.append(nn.ReLU())
            in_dim = hidden_size
        self.backbone = nn.Sequential(*module_list)

        if self.args.feat == 'b':
            self.feat = nn.Sequential(
                nn.BatchNorm1d(in_dim, affine=False)
                )
        elif self.args.feat == 'bf':
            self.feat = nn.Sequential(
                nn.BatchNorm1d(in_dim, affine=False), 
                nn.Linear(in_dim, in_dim)
                )
        elif self.args.feat == 'f':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim)
            )
        elif self.args.feat == 'fbg':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.BatchNorm1d(in_dim, affine=True),
                nn.GELU()
            )
        elif self.args.feat == 'fg':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU()
            )
        elif self.args.feat == 'ft':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.Tanh()
            )
        elif self.args.feat == 'fp':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.PReLU()
            )
        elif self.args.feat == 'fr':
            self.feat = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU()
            )
        else:
            self.feat = nn.Sequential(
                nn.Identity()
            )

        self.fc = nn.Linear(in_dim, out_dim, bias=args.bias)
        if self.args.init_s > 0 and self.args.init_s != 1:
            self.fc.weight.data = self.fc.weight.data * self.args.init_s
            print('---rescale fc weight---')

        self.max_action = max_action

    def forward(self, state, ret_feat=False):
        x = self.backbone(state)
        feat = self.feat(x)
        out = self.fc(feat)
        if ret_feat:
            return out, feat
        else:
            return out

    def forward_feat(self, x):
        out1 = self.backbone[0:3](x)
        out2 = self.backbone[3:6](out1)
        out3 = self.backbone[6:9](out2)
        return out1, out2, out3

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # Modified: Clip the actions, since we do not have a tanh in the actor.
        action = self(state).clamp(min=-self.max_action, max=self.max_action)
        return action.cpu().data.numpy().flatten()
