import os
import torch
import wandb
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from dataset import get_dataloader
from models.model import RegressionResNet, MLP, VGG, LeNet
from utils.train_utils import get_feat_pred, get_scheduler, get_theoretical_solution, compute_metrics
from utils.utils import set_log_path, log, print_args


def train_one_epoch(model, data_loader, optimizer, criterion, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    all_feats = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device, non_blocking=True), targets.float().to(device, non_blocking=True)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        optimizer.zero_grad()

        if images.ndim == 4 and model.args.arch.startswith('mlp'):
            images = images.view(images.shape[0], -1)
        outputs, feats = model(images, ret_feat=True)
        all_feats.append(feats.data)

        loss = criterion(outputs, targets)
        if args.ufm:
            l2reg_H = torch.sum(feats ** 2) * args.lambda_H / args.batch_size
            l2reg_W = torch.sum(model.fc.weight ** 2) * args.lambda_W
            loss = loss + l2reg_H + l2reg_W

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_train_loss = running_loss / len(data_loader)

    all_feats = torch.cat(all_feats)
    return all_feats, running_train_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('---- start loading data  ----')
    if args.dataset in ['carla', 'carla2d']: 
        args.num_y = 2
    elif args.dataset in ['carla1d']:
        args.num_y = 1
    train_loader, val_loader = get_dataloader(args)
    print('---- finish loading data  ----')
    if args.dataset in ['swimmer', 'reacher', 'hopper', 'reacher_ab', 'swimmer_ab']:
        args.num_x = train_loader.dataset.state_dim
        if args.which_y == -1:
            args.num_y = train_loader.dataset.action_dim
        else:
            args.num_y = 1
    elif args.dataset in ['mnist']:
        args.num_x = 28 ** 2
        args.num_y = 1
    elif args.dataset in ['cifar10']:
        args.num_x = 32 * 32 * 3
        args.num_y = 1
    print_args(args)
    # ================== setup wandb  ==================

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    os.environ["WANDB_ARTIFACT_DIR"] = "/scratch/lg154/sseg/wandb"
    os.environ["WANDB_DATA_DIR"] = "/scratch/lg154/sseg/wandb/data"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='NRC_final',  # + args.dataset,
               name=args.exp_name.split('/')[-1]
               )
    wandb.config.update(args)

    # =================== theoretical solution ================
    if args.dataset in ['swimmer', 'reacher', 'hopper']:
        theory_stat = train_loader.dataset.get_theory_stats(center=args.bias)
    else:
        theory_stat = get_theoretical_solution(train_loader, args, center=args.bias)
    print(',\n'.join([ f'{key}: {value}' for key, value in theory_stat.items()]))

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
    criterion = nn.MSELoss()
    wandb.watch(model, criterion, log="all", log_freq=20)
    for epoch in range(args.start_epoch, args.max_epoch):

        if epoch == 0 or epoch % args.log_freq == 0:
            # === cosine between Wi's
            W = model.fc.weight.data  # [2, 512]
            WWT = (W @ W.T).cpu().numpy()

            # ===============compute train mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, train_loader)
            if args.dataset in ['mnist', 'cifar10']: labels = labels.float()
            torch.cuda.empty_cache()
            if args.y_norm not in ['null', 'n']:
                y_shift, std = torch.tensor(train_loader.dataset.y_shift).to(preds.device), torch.tensor(
                    train_loader.dataset.std).to(preds.device)
                preds = preds @ std + y_shift
                labels = labels @ std + y_shift
            if preds.shape[-1] == 2:
                train_loss0 = torch.sum((preds[:, 0] - labels[:, 0]) ** 2) / preds.shape[0]
                train_loss1 = torch.sum((preds[:, 1] - labels[:, 1]) ** 2) / preds.shape[0]
            train_loss = torch.mean((preds - labels) ** 2)

            nc_train = compute_metrics(W, all_feats)
            train_hnorm = torch.norm(all_feats, p=2, dim=1).mean().item()
            train_wnorm = torch.norm(W, p=2, dim=1).mean().item()

            # ===============compute val mse and projection error==================
            all_feats, preds, labels = get_feat_pred(model, val_loader)
            if args.dataset in ['mnist', 'cifar10']: labels = labels.float()
            torch.cuda.empty_cache()
            if args.y_norm not in ['null', 'n']:
                y_shift, std = torch.tensor(train_loader.dataset.y_shift).to(preds.device), torch.tensor(
                    train_loader.dataset.std).to(preds.device)
                preds = preds @ std + y_shift
                labels = labels @ std + y_shift
            if preds.shape[-1] == 2:
                val_loss0 = torch.sum((preds[:, 0] - labels[:, 0]) ** 2) / preds.shape[0]
                val_loss1 = torch.sum((preds[:, 1] - labels[:, 1]) ** 2) / preds.shape[0]
            val_loss = torch.mean((preds - labels) ** 2)

            nc_val = compute_metrics(W, all_feats)
            val_hnorm = torch.norm(all_feats, p=2, dim=1).mean().item()
            del all_feats, preds, labels
            torch.cuda.empty_cache()

            # ================ NC2 ================
            if args.num_y >= 2:
                WWT_normalized = WWT / np.linalg.norm(WWT)
                min_eigval = theory_stat['min_eigval']
                Sigma_sqrt = theory_stat['Sigma_sqrt']
                W_outer = args.lambda_H * (Sigma_sqrt / np.sqrt(args.lambda_H * args.lambda_W) - np.eye(args.num_y))

                c_to_plot = np.linspace(0, min_eigval, num=1000)
                NC3_to_plot = []
                for c in c_to_plot:
                    c_sqrt = c ** 0.5
                    A = Sigma_sqrt - c_sqrt * np.eye(Sigma_sqrt.shape[0])
                    A_normalized = A / np.linalg.norm(A)
                    diff_mat = WWT_normalized - A_normalized
                    NC3_to_plot.append(np.linalg.norm(diff_mat))

                data = [[a, b] for (a, b) in zip(c_to_plot, NC3_to_plot)]
                table = wandb.Table(data=data, columns=["c", "NC3"])
                wandb.log(
                    {"NC3(c)": wandb.plot.line(table, "c", "NC3", title="NC3 as a Function of c")},
                    step=epoch)
                best_c = c_to_plot[np.argmin(NC3_to_plot)]
                NC3 = min(NC3_to_plot)
            else:
                best_c, NC3 = 0, 0

            # ================ log to wandb ================
            nc_dt = {
                'train/train_nc1': nc_train['nc1'],
                'train/train_nc1n': nc_train['nc1n'],
                'train/train_nc2': nc_train['nc2'],
                'train/train_nc2n': nc_train['nc2n'],
                'train/train_nc3': NC3,
                'train/best_c': best_c,

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

                'NC1/nc1_pc1': nc_train['nc1_pc1'],
                'NC1/nc1_pc2': nc_train['nc1_pc2'],
                'NC1/nc1_pc3': nc_train['nc1_pc3'],
                'NC1/nc1_pc4': nc_train['nc1_pc4'],
                'NC1/nc1_pc5': nc_train['nc1_pc5'],

                'val/val_nc1': nc_val['nc1'],
                'val/val_nc1n': nc_val['nc1n'],
                'val/val_nc2': nc_val['nc2'],
                'val/val_nc2n': nc_val['nc2n'],

                'other/lr': optimizer.param_groups[0]['lr'],
                'other/train_hnorm': train_hnorm,
                'other/val_hnorm': val_hnorm,
                'other/wnorm': train_wnorm,

                'mse/train_mse': train_loss,
                'mse/val_mse': val_loss,
            }

            wandb.log(nc_dt, step=epoch)
            
            print('---- finish logging  ----')

        # =============== store model ==================
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
        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()
        all_feats, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args=args)
        # =============== save w
        if args.save_w == 't' and (epoch + 1) % 100 == 0:
            import pickle
            with open(os.path.join(args.save_dir, 'fc_w.pkl'), 'wb') as f:
                pickle.dump({'w': model.fc.weight.data.cpu().numpy(), 'wwt': W_outer, 'Sigma_sqrt': Sigma_sqrt}, f)


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
    parser.add_argument('--dataset', type=str, default='Carla')
    parser.add_argument('--data_ratio', type=float, default=1.0)

    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--num_y', type=int, default=2)
    parser.add_argument('--which_y', type=int, default=-1)
    parser.add_argument('--y_norm', type=str, default='null')
    parser.add_argument('--x_norm', type=str, default='null')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--w', type=str, default='null')
    parser.add_argument('--bn', type=str, default='f')  # f|t|p false|true|parametric
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
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--log_freq', default=5, type=int)

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

    set_seed(args.seed)
    main(args)
