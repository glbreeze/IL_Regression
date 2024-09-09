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
            
        self.backbone = nn.Sequential(nn.Sequential(resnet_model.conv1, resnet_model.bn1, resnet_model.relu, resnet_model.maxpool),
                                      resnet_model.layer1,
                                      resnet_model.layer2,
                                      resnet_model.layer3,
                                      resnet_model.layer4,
                                      )
        
        if self.args.feat == 'f':
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
    
    def forward(self, x, ret_feat=False):
        x = self.backbone(x)
        feat = self.feat(x)
        out = self.fc(feat)
        if ret_feat:
            return out, feat
        else:
            return out

    def forward_feat(self, x):
        feat_list = []
        for m in self.backbone:
            x = m(x)
            feat_list.append(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1))
        return feat_list


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, args, arch='256_256', max_action=1.0):
        super(MLP, self).__init__()
        self.args = args

        # ====== backbone ======
        module_list = []
        for i, hidden_size in enumerate(arch.split('_')):
            hidden_size = int(hidden_size)

            if args.drop >= 0:
                dropout_layer = nn.Dropout(args.drop)
            else:
                dropout_layer = nn.Identity()

            if args.bn == 't':
                norm_layer = nn.BatchNorm1d(hidden_size, affine=False)
            elif args.bn == 'p':
                norm_layer = nn.BatchNorm1d(hidden_size, affine=True)
            else:
                norm_layer = nn.Identity()

            if args.act == 'elu':
                activation = nn.ELU()
            elif args.act == 'lrelu':
                activation = nn.LeakyReLU(0.1)
            else:
                activation = nn.ReLU()

            module_list.append(nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                dropout_layer,
                norm_layer,
                activation
            ))
            in_dim = hidden_size

        self.backbone = nn.Sequential(*module_list)

        if self.args.feat == 'b':
            self.feat = nn.BatchNorm1d(in_dim, affine=False)
        elif self.args.feat == 'f':
            self.feat = nn.Linear(in_dim, in_dim)
        else:
            self.feat = nn.Identity()

        self.fc = nn.Linear(in_dim, out_dim, bias=args.bias)
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
        feat_list = []
        for m in self.backbone:
            x = m(x)
            feat_list.append(x)
        return feat_list

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        # Modified: Clip the actions, since we do not have a tanh in the actor.
        action = self(state).clamp(min=-self.max_action, max=self.max_action)
        return action.cpu().data.numpy().flatten()
