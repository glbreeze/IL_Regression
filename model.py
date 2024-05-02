import torch
import torch.nn as nn
from torchvision import models

class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2, args=None):
        super(RegressionResNet, self).__init__()
        self.args = args 

        if args.arch == 'resnet18' or args.arch == 'res18':
            resnet_model = models.resnet18(pretrained=pretrained)
        elif args.arch == 'resnet50' or args.arch == 'res50':
            resnet_model = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-2])
        
        if self.args.feat == 'b': 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
                nn.Flatten(),
                nn.BatchNorm1d(resnet_model.fc.in_features, affine=False)
                )
        elif self.args.feat == 'bf': 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(resnet_model.fc.in_features, affine=False), 
                nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features)
                )
        elif self.args.feat == 'bft': 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(resnet_model.fc.in_features, affine=False), 
                nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features), 
                nn.Tanh()
                )
        elif self.args.feat == 'bfrf': 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(resnet_model.fc.in_features, affine=False), 
                nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features), 
                nn.ReLU(), 
                nn.Linear(resnet_model.fc.in_features, resnet_model.fc.in_features), 
                )
        elif self.args.feat == 'f':    # without GAP 
            self.feat = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(resnet_model.fc.in_features * 7 * 7, resnet_model.fc.in_features)
            )
        elif self.args.feat == 'ft':    # without GAP 
            self.feat = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(resnet_model.fc.in_features * 7 * 7, resnet_model.fc.in_features),
                nn.Tanh()
            )
        else: 
            self.feat = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
                nn.Flatten()
            )

        self.fc = nn.Linear(resnet_model.fc.in_features, num_outputs, bias=args.bias)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x, ret_feat=False):
        x = self.backbone(x)
        feat = self.feat(x)
        out = self.fc(feat)
        if ret_feat:
            return out, feat
        else:
            return out

    def get_last_layer_embeddings(self, x):
        """Extract embeddings from the last layer using a hook and global average pooling."""
        def hook_fn(module, input, output):
            pooled_output = self.global_avg_pool(output)
            self.embeddings = pooled_output.view(pooled_output.size(0), -1).detach()

        hook = self.model.layer4.register_forward_hook(hook_fn)
        self.forward(x)
        hook.remove()
        return self.embeddings
