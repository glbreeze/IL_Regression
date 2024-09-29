import os
import pdb
import torch
import wandb
import random
import pickle
from scipy.linalg import qr
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from sklearn.decomposition import PCA

from dataset import SubDataset, get_dataloader
from model import RegressionResNet, MLP, VGG, LeNet
from train_utils import get_feat_pred, gram_schmidt, get_scheduler, get_theoretical_solution, compute_metrics, \
    get_all_feat, plot_var_ratio_hw, analysis_feat, plot_var_ratio
from utils import print_model_param_nums, set_log_path, log, print_args, matrix_with_angle


def train_one_epoch(model, data_loader, optimizer, criterion, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    num_acc = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if images.ndim == 4 and model.args.arch.startswith('mlp'):
            images = images.view(images.shape[0], -1)
        outputs, feats = model(images, ret_feat=True)

        loss = criterion(outputs, targets)
        if args.ufm:
            l2reg_H = torch.sum(feats ** 2) * args.lambda_H / args.batch_size
            l2reg_W = torch.sum(model.fc.weight ** 2) * args.lambda_W
            loss = loss + l2reg_H + l2reg_W

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_acc += (outputs.argmax(dim=-1) == targets).sum().item()
    running_train_loss = running_loss / len(data_loader)
    running_train_acc = num_acc / len(data_loader.dataset)

    return running_train_loss, running_train_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloader(args)
    if args.dataset == 'mnist':
        args.num_y = 10
        args.num_x = 28 * 28
    elif args.dataset in ['reacher', 'swimmer', 'hopper']:
        args.num_x = train_loader.dataset.state_dim
        args.num_y = 5
    elif args.dataset == 'cifar10':
        args.num_y = 10
        args.num_x = 32 * 32 * 3

    # ================== setup wandb  ==================

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    os.environ["WANDB_ARTIFACT_DIR"] = "/scratch/lg154/sseg/wandb"
    os.environ["WANDB_DATA_DIR"] = "/scratch/lg154/sseg/wandb/data"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='NRC_sep',  # + args.dataset,
               name=args.exp_name.split('/')[-1],
               settings=wandb.Settings(start_method="fork")
               )
    wandb.config.update(args)

    # ===================    Load model   ===================
    if args.arch.startswith('res'):
        model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
    elif args.arch.startswith('mlp'):
        model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp', ''))
    elif args.arch.startswith('vgg'):
        model = VGG(num_classes=args.num_y, args=args)
    elif args.arch.startswith('le'):
        model = LeNet(num_classes=args.num_y, args=args)
    if torch.cuda.is_available():
        model = model.cuda()

    num_params = sum([param.nelement() for param in model.parameters()])
    log('--- total num of params: {} ---'.format(num_params))

    if model.fc.bias is not None:
        log("--- classification layer has bias terms. ---")
    else:
        log("--- classification layer DO NOT have bias terms. ---")

    # ==== optimizer and scheduler
    if args.ufm:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(args, optimizer)
    if args.warmup > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ================== Training ==================
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all", log_freq=20)
    for epoch in range(args.start_epoch, args.max_epoch):

        if epoch == 0 or epoch % args.log_freq == 0:
            # === cosine between Wi's
            W = model.fc.weight.data  # [2, 512]

            # ===============compute train mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, train_loader)
            train_acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean().item()
            nc_train = compute_metrics(W, all_feats)
            nc_cls = analysis_feat(labels.squeeze(), all_feats, num_classes=args.num_y, W=W)

            # ===============compute val mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, val_loader)
            val_acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean().item()
            nc_val = compute_metrics(W, all_feats)

            del all_feats, preds, labels
            torch.cuda.empty_cache()

            # ================ log to wandb ================
            nc_dt = {
                'train/train_nc1': nc_train['nc1'],
                'train/train_nc1n': nc_train['nc1n'],
                'train/train_nc2': nc_train['nc2'],
                'train/train_nc2n': nc_train['nc2n'],

                'EVR/EVR1': nc_train['EVR1'],
                'EVR/EVR2': nc_train['EVR2'],
                'EVR/EVR3': nc_train['EVR3'],
                'EVR/EVR4': nc_train['EVR4'],
                'EVR/EVR5': nc_train['EVR5'],

                'CVR/CVR5': nc_train['CVR5'],
                'CVR/CVR10': nc_train['CVR10'],
                'CVR/CVR15': nc_train['CVR15'],
                'CVR/CVR20': nc_train['CVR20'],
                'CVR/CVR30': nc_train['CVR30'],

                'val/val_nc1': nc_val['nc1'],
                'val/val_nc1n': nc_val['nc1n'],
                'val/val_nc2': nc_val['nc2'],
                'val/val_nc2n': nc_val['nc2n'],

                'other/lr': optimizer.param_groups[0]['lr'],

                'cls_nc/nc1': nc_cls['nc1'],
                'cls_nc/nc2': nc_cls['nc2'],
                'cls_nc/nc3': nc_cls['nc3'],
                'cls_nc/train_acc': train_acc,
                'cls_nc/val_acc': val_acc,
            }
            wandb.log(nc_dt, step=epoch)

        # ================ log the figure ================
        if epoch == 0 or (epoch + 1) % (args.max_epoch // 20) == 0:
            def get_rank(m_by_layer):
                vr_by_layer = {}
                rk_by_layer = {}
                for layer_id, m in m_by_layer.items():
                    cov = m.T @ m
                    U, S, Vt = np.linalg.svd(cov)
                    S = S[: min(m.shape[0], m.shape[1])]

                    s_ratio = S / np.sum(S)
                    entropy = -np.sum(s_ratio * np.log(s_ratio + 1e-12))
                    effective_rank = np.exp(entropy)

                    vr_by_layer[layer_id] = s_ratio
                    rk_by_layer[layer_id] = effective_rank
                return vr_by_layer, rk_by_layer

            include_input = False if args.dataset in ['mnist', 'cifar10', 'cifar100'] else True
            feat_by_layer = get_all_feat(model, train_loader, include_input=include_input)

            weight_by_layer = {id: model.backbone[id][0] for id in range(len(model.backbone))}
            for id in weight_by_layer.keys():
                layer = weight_by_layer[id]
                weight = layer.weight.data
                if isinstance(layer, nn.Linear):
                    weight_by_layer[id] = weight.cpu().numpy()
                else:
                    weight_by_layer[id] = weight.view(weight.size(0), -1).cpu().numpy()

            vr_feat, rk_feat = get_rank(feat_by_layer)
            vr_weight, rk_weight = get_rank(weight_by_layer)

            fig = plot_var_ratio_hw(vr_feat, vr_weight)
            wandb.log({"chart": wandb.Image(fig)}, step=epoch)

            wandb.log({f'feat_rank/{id}': rk for id, rk in rk_feat.items()}, step=epoch)
            wandb.log({f'weight_rank/{id}': rk for id, rk in rk_weight.items()}, step=epoch)

        if (epoch == 0 or (epoch + 1) % args.save_freq == 0) and args.save_freq > 0:
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

            log('--save model to {}'.format(ckpt_path))

        # =============== train model ==================
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, args=args)
        if epoch % 5 == 0:
            wandb.log({'cls_nc/train_loss': train_loss, 'cls_nc/run_acc': train_acc}, step=epoch)
        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regression NC")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--cls', action='store_true', default=False)

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--num_y', type=int, default=2)
    parser.add_argument('--y_norm', type=str, default='null')
    parser.add_argument('--x_norm', type=str, default='null')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--w', type=str, default='null')
    parser.add_argument('--bn', type=str, default='p')  # f|t|p false|true|parametric
    parser.add_argument('--drop', type=float, default='0')

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--feat', type=str, default='null')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lambda_H', type=float, default=1e-3)
    parser.add_argument('--lambda_W', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='multi_step')
    parser.add_argument('--save_w', type=str, default='f')

    parser.add_argument('--ufm', default=False, action='store_true')
    parser.add_argument('--bias', default=False, action='store_true')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_freq', default=-1, type=int)
    parser.add_argument('--log_freq', default=10, type=int)

    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--exp_name', type=str, default='exp')
    args = parser.parse_args()

    args.save_dir = os.path.join("./result/{}/".format(args.dataset), args.exp_name)
    if args.resume is not None:
        args.resume = os.path.join('./result', args.resume)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    set_log_path(args.save_dir)
    log('save log to path {}'.format(args.save_dir))
    log(print_args(args))

    set_seed(args.seed)
    main(args)
