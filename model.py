import torch
import torch.nn as nn
from torchvision import models

class RegressionResNet(nn.Module):
    def __init__(self, pretrained=True, num_outputs=2, bias=True):
        super(RegressionResNet, self).__init__()
        resnet_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet_model.children())[:-1])

        self.fc = nn.Linear(resnet_model.fc.in_features, num_outputs, bias=bias)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x, ret_feat=False):
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
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
