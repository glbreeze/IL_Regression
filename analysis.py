import os
import torch
import pickle
import numpy as np
import argparse
from dataset import get_dataloader
from model import RegressionResNet, MLP
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Regression NC")
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--data_ratio', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--feat', type=str, default='null')
parser.add_argument('--num_y', type=int, default=1)
parser.add_argument('--which_y', type=int, default=-1)
parser.add_argument('--y_norm', type=str, default='null')
parser.add_argument('--x_norm', type=str, default='null')
parser.add_argument('--bn', type=str, default='p')  # f|t|p false|true|parametric
parser.add_argument('--init_s', type=float, default='1.0')

parser.add_argument('--ufm', default=False, action='store_true')
parser.add_argument('--bias', default=False, action='store_true')

parser.add_argument('--exp_name', type=str, default='exp')
args = parser.parse_args([])
args.act = 'relu'
args.exp_dir = os.path.join(f'result/{args.dataset}', args.exp_name)

# args.exp_name = 'mn_res18_WD1e-3_LR1e-3'
if args.dataset == 'reacher':
    args.data_ratio = 0.1
    EP=999
    args.exp_name = 'dr0.1_MLP256.256.256_WD1e-2'
elif args.dataset == 'swimmer':
    args.data_ratio = 0.01
    EP = 999
    args.exp_name = 'sw0.01_mlp3_WD1e-2_LR1e-2'
elif args.dataset == 'mnist':
    EP = 499

# ======= data loader
train_loader, val_loader = get_dataloader(args)
if args.dataset in ['swimmer', 'reacher', 'hopper', 'reacher_ab', 'swimmer_ab']:
    args.num_x = train_loader.dataset.state_dim
    if args.which_y == -1:
        args.num_y = train_loader.dataset.action_dim
    else:
        args.num_y = 1
elif args.dataset in ['mnist']:
    args.num_y = 1

# ====== load model
if args.arch.startswith('res'):
    model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
elif args.arch.startswith('mlp'):
    model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp', ''))
if torch.cuda.is_available():
    model = model.cuda()

state_dict = torch.load(os.path.join(args.exp_dir, f'ep{EP}_ckpt.pth'), map_location=device)
model.load_state_dict(state_dict['model_state_dict'])

# ======= get feat
model.eval()
all_feats1, all_feats2, all_feats3 = [], [], []

with torch.no_grad():
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        feats = model.forward_feat(input)

        all_feats1.append(feats[0])
        all_feats2.append(feats[1])
        all_feats3.append(feats[2])

    all_feats1 = torch.cat(all_feats1)
    all_feats2 = torch.cat(all_feats2)
    all_feats3 = torch.cat(all_feats3)

all_feats1, all_feats2, all_feats3 = all_feats1.cpu().numpy(), all_feats2.cpu().numpy(), all_feats3.cpu().numpy()

import pickle
with open(os.path.join(args.exp_dir, 'feat_analysis.pkl'), 'wb') as f:
    pickle.dump([all_feats1, all_feats2, all_feats3], f)

var_ratios, var_ratios1 = [], []
for feat in [all_feats1, all_feats2, all_feats3]:
    cov = feat.T @ feat / len(feat)
    U, S, Vt = np.linalg.svd(cov)
    var_ratio = np.cumsum(S)/np.sum(S)
    var_ratios.append(var_ratio)

    # pca_feat = PCA(n_components=feat.shape[1])
    # pca_feat.fit(feat)
    # var_ratio1 = np.cumsum(pca_feat.explained_variance_ratio_) / np.sum(pca_feat.explained_variance_ratio_)


for i in range(3):
    plt.plot(np.arange(len(S)), var_ratios[i], label=f'cumulative explained var for layer -{i}')
plt.legend()
plt.title(f'Cumulative explained var for {args.dataset} WD 1e-2')



