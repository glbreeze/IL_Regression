import os
import pdb
import pickle
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from sklearn.decomposition import PCA


from dataset import SubDataset, get_dataloader
from model import RegressionResNet, MLP
from train_utils import Graph_Vars, get_feat_pred, gram_schmidt, get_scheduler, Train_Vars, get_theoretical_solution, compute_metrics
from utils import print_model_param_nums, set_log_path, log, print_args


def train_one_epoch(model, data_loader, optimizer, criterion, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    all_feats = []
    for batch_idx, batch in enumerate(data_loader):
        images = batch['input'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs, feats = model(images, ret_feat=True)
        all_feats.append(feats.data)

        loss = criterion(outputs, targets)
        if args.ufm:
            l2reg_H = torch.sum(feats**2) * args.lambda_H / args.batch_size
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

    train_loader, val_loader = get_dataloader(args)
    if args.dataset == 'swimmer' or args.dataset == 'reacher':
        args.num_x = train_loader.dataset.state_dim
        args.num_y = train_loader.dataset.action_dim

    if args.arch.startswith('res'):
        model = RegressionResNet(pretrained=True, num_outputs=args.num_y, args=args).to(device)
    elif args.arch.startswith('mlp'):
        model = MLP(in_dim=args.num_x, out_dim=args.num_y, args=args, arch=args.arch.replace('mlp',''))
    _ = print_model_param_nums(model=model)
    if torch.cuda.is_available():
        model = model.cuda()
    if model.fc.bias is not None:
        log("classification layer has bias terms.")
    else:
        log("classification layer DO NOT have bias terms.")

    if args.ufm:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler(args, optimizer)
    if args.warmup>0: 
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup)
        
    # lambda0 = lambda epoch: epoch / args.warmup if epoch < args.warmup else 1 * 0.2**((epoch-800)//100)

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

    # =================== theoretical solution ================
    W_outer, Sigma_sqrt, all_labels, theory_stat = get_theoretical_solution(train_loader, args, bias=None, all_labels=None, center=args.bias)
    if args.dataset in ['swimmer', 'reacher']: 
        theory_stat = train_loader.dataset.get_theory_stats(center=args.bias)

    # ================== setup wandb  ==================
    args.s00 = Sigma_sqrt[0, 0]
    args.s01 = Sigma_sqrt[0, 1]
    args.s11 = Sigma_sqrt[1, 1]

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    os.environ["WANDB_ARTIFACT_DIR"] = "/scratch/lg154/sseg/wandb"
    os.environ["WANDB_DATA_DIR"] = "/scratch/lg154/sseg/wandb/data"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='reg1_' + args.dataset,
               name=args.exp_name.split('/')[-1]
               )
    wandb.config.update(args)

    # ================== log the theoretical result  ==================
    filename = os.path.join(args.save_dir, 'theory.pkl')
    with open(filename, 'wb') as f:
        pickle.dump({'target':all_labels.cpu().numpy(), 'W_outer':W_outer, 'lambda_H':args.lambda_H, 'lambda_W':args.lambda_W}, f)
        log('--store theoretical result to {}'.format(filename))
    log('====> Theoretical s00: {:.5f}, s01: {:.5f}, s11: {:.5f}'.format(Sigma_sqrt[0, 0], Sigma_sqrt[0, 1], Sigma_sqrt[1, 1]))
    log('====> Theoretical ww00: {:.4f}, ww01: {:.4f}, ww11: {:.4f}'.format(W_outer[0, 0], W_outer[0, 1], W_outer[1, 1]))
    log(', '.join([f'{key}: {round(value,4)}' for key, value in theory_stat.items()]))

    # ================== Training ==================
    criterion = nn.MSELoss()
    nc_tracker = Graph_Vars(dim=args.num_y)
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(args.start_epoch, args.max_epoch):
        
        # === cosine between Wi's
        W = model.fc.weight.data  # [2, 512]
        WWT = (W @ W.T).cpu().numpy()

        # ==== calculate training feature with updated W
        all_feats, preds, labels = get_feat_pred(model, train_loader)
        train_loss = criterion(preds, labels)
        nc_train = compute_metrics(W, all_feats)

        # ===============compute val mse and projection error==================
        all_feats, preds, labels = get_feat_pred(model, val_loader)
        val_loss = criterion(preds, labels)
        nc_val = compute_metrics(W, all_feats)

        nc_dt = {
            'ww00': WWT[0, 0].item(),
            'ww01': WWT[0, 1].item(),
            'ww11': WWT[1, 1].item(),
            'w_cos': F.cosine_similarity(W[0], W[1], dim=0).item(),
            'train_mse': train_loss,
            'train_nc1': nc_train['nc1'],
            'train_nc3': nc_train['nc3'],
            'train_nc3a': nc_train['nc3a'],
            'val_mse': val_loss,
            'val_nc1': nc_val['nc1'],
            'val_nc3': nc_val['nc3'],
            'val_nc3a': nc_val['nc3a'],
            'h_norm': torch.norm(all_feats, p=2, dim=1).mean().item()
        }

        # ================ NC2 ================
        WWT_normalized = WWT / np.linalg.norm(WWT)
        min_eigval = theory_stat['min_eigval']
        Sigma_sqrt = np.array([theory_stat[k] for k in ['sigma11', 'sigma12', 'sigma21', 'sigma22']]).reshape(2, 2)

        c_to_plot = np.linspace(0, min_eigval, num=1000)
        NC2_to_plot = []
        NC2_11_to_plot = []
        NC2_12_to_plot = []
        NC2_22_to_plot = []
        for c in c_to_plot:
            c_sqrt = c ** 0.5
            A = Sigma_sqrt - c_sqrt * np.eye(2)
            A_normalized = A / np.linalg.norm(A)
            diff_mat = WWT_normalized - A_normalized
            NC2_to_plot.append(np.linalg.norm(diff_mat))
            NC2_11_to_plot.append(diff_mat[0, 0])
            NC2_12_to_plot.append(diff_mat[0, 1])
            NC2_22_to_plot.append(diff_mat[1, 1])

        data = [[a, b, c, d, f] for (a, b, c, d, f) in
                zip(c_to_plot, NC2_to_plot, NC2_11_to_plot, NC2_12_to_plot, NC2_22_to_plot)]
        table = wandb.Table(data=data, columns=["c", "NC2", "NC2_11", "NC2_12", "NC2_22"])
        wandb.log(
            {
                "NC2(c)": wandb.plot.line(
                    table, "c", "NC2", title="NC2 as a Function of c"
                )
            }, step=epoch
        )

        nc_dt['nc2'] = min(NC2_to_plot)
        nc_tracker.load_dt(nc_dt, epoch=epoch)

        wandb.log(
            {'train/lr': optimizer.param_groups[0]['lr'],
             'train/train_mse': train_loss,
             'train/train_nc1': nc_dt['train_nc1'],
             'train/train_nc3': nc_dt['train_nc3'],
             'train/train_nc3a': nc_dt['train_nc3a'],

             'val/val_mse': val_loss,
             'val/val_nc1': nc_dt['val_nc1'],
             'val/val_nc3': nc_dt['val_nc3'],
             'val/val_nc3a': nc_dt['val_nc3a'],

             'W/ww00': nc_dt['ww00'],
             'W/ww01': nc_dt['ww01'],
             'W/ww11': nc_dt['ww11'],
             'W/w_cos': nc_dt['w_cos'],
             'W/nc2': nc_dt['nc2'],
             'W/h_norm': nc_dt['h_norm'],
             
             'NC2/ww00_d': abs(nc_dt['ww00'] - W_outer[0, 0])/(abs(W_outer[0, 0])+1e-8),
             'NC2/ww01_d': abs(nc_dt['ww01'] - W_outer[0, 1])/(abs(W_outer[0, 1])+1e-8),
             'NC2/ww11_d': abs(nc_dt['ww11'] - W_outer[1, 1])/(abs(W_outer[1, 1])+1e-8),
             'NC2/ww_d': np.sum( (WWT/np.linalg.norm(WWT) -W_outer/np.linalg.norm(W_outer))**2 ),
             'NC2/ww_d1': np.sum( ((WWT -W_outer)/np.linalg.norm(W_outer))**2 )
             },
            step=epoch)

        log('Epoch {}/{}, runnning train mse: {:.4f}, ww00: {:.4f}, ww01: {:.4f}, ww11: {:.4f}'.format(
            epoch, args.max_epoch, train_loss, nc_dt['ww00'], nc_dt['ww01'], nc_dt['ww11']
        ))

        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

            log('--save model to {}'.format(ckpt_path))
        
        if (epoch+1) % (args.save_freq*10) == 0:
            filename = os.path.join(args.save_dir, 'train_nc{}.pkl'.format(epoch))
            with open(filename, 'wb') as f:
                pickle.dump(nc_tracker, f)
            
        all_feats, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args=args)
        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Regression NC")
    parser.add_argument('--dataset', type=str, default='Carla')
    parser.add_argument('--data_ratio', type=float, default=1.0)
    
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--y_norm', type=str, default='null')

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_y', type=int, default=2)
    parser.add_argument('--feat', type=str, default='b')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--lambda_H', type=float, default=1e-3)
    parser.add_argument('--lambda_W', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='multi_step')

    parser.add_argument('--ufm', default=False, action='store_true')
    parser.add_argument('--bias', default=False, action='store_true')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_freq', default=10, type=int)

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

    main(args)
