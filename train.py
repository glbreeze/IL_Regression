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


from dataset import SubDataset, get_dataloader
from model import RegressionResNet, MLP
from train_utils import Graph_Vars, get_feat_pred, compute_cosine_norm, gram_schmidt, get_scheduler, Train_Vars, get_theoretical_solution
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

    # val_dataset = ImageTargetDataset('/vast/zz4330/Carla_JPG/Val/images', '/vast/zz4330/Carla_JPG/Val/targets', transform=transform, dim=args.num_y)
    # val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
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
            
        all_feats, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args=args)
        if epoch < args.warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # === cosine between Wi's
        W = model.fc.weight.data  # [2, 512]
        W_outer_pred = torch.matmul(W, W.T).cpu().numpy()
        W_outer_d = np.sum(abs(W_outer - W_outer_pred))
  
        # ==== calculate training feature with updated W
        if True:
            all_feats, preds, labels = get_feat_pred(model, train_loader)
            train_loss = criterion(preds, labels)

        # === compute projection error
        U = gram_schmidt(W)
        P_E = torch.mm(U.T, U)  # Projection matrix using orthonormal basis
        h_projected = torch.mm(all_feats, P_E)
        projection_error_train = mse_loss(h_projected, all_feats).item()

        # === compute theoretical value for WW^T with updated bias
        if args.bias and args.ufm:
            W_outer_new, _, _ = get_theoretical_solution(train_loader, args, all_labels=labels, bias=model.fc.bias.data)
            W_outer_d = np.sum(abs(W_outer_new - W_outer_pred))
            wandb.log(
                {'W/ww00_th1': W_outer_new[0,0],
                 'W/ww01_th1': W_outer_new[0,1],
                 'W/ww11_th1': W_outer_new[1,1], 
                 'W/b0': model.fc.bias[0].item(), 
                 'W/b1': model.fc.bias[1].item(),
                 },
                step=epoch)
        # ================ NC2 ================
        WWT = W_outer_pred
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

        slamH_to_plot = np.linspace(0.0001, min_eigval/np.sqrt(args.wd), num=1000)
        NC2_to_plot = []
        NC2_norm_to_plot = []
        for slamH in slamH_to_plot:
            A = slamH * Sigma_sqrt / (args.wd ** 0.5) - (slamH ** 2) * np.eye(2)
            NC2_to_plot.append(np.linalg.norm(WWT - A))
            NC2_norm_to_plot.append(np.linalg.norm(WWT_normalized - A / np.maximum(np.linalg.norm(A), 1e-6)))

        data = [[a, b, c] for (a, b, c) in zip(slamH_to_plot, NC2_to_plot, NC2_norm_to_plot)]
        table = wandb.Table(data=data, columns=["slamH", "NC2", "NC2_norm"])
        # wandb.log(
        #     {
        #         "NC2(slamH)": wandb.plot.line(
        #             table, "slamH", "NC2", title="NC2 as a Function of c and lamH"
        #         )
        #     }
        # )
        wandb.log(
            {
                "NC2(slamH)": wandb.plot.line(
                    table, "slamH", "NC2_norm", title="NC2 norm as a Function of slamH"
                )
            }, step=epoch
        )
        best_slamH = slamH_to_plot[np.argmin(np.array(NC2_to_plot))]
        wandb.log({'slamH': best_slamH},step=epoch)

        # ===============compute val mse and projection error==================
        if args.dataset in ['swimmer', 'reacher']: 
            feats, preds, labels = get_feat_pred(model, val_loader)
            val_loss = criterion(preds, labels)

            h_projected = torch.mm(feats, P_E)
            projection_error_val = mse_loss(h_projected, feats).item()
        else: 
            val_loss = 0 
            projection_error_val = 0

        nc_dt = {'lr': optimizer.param_groups[0]['lr'],
                 'train_mse': train_loss,
                 'train_proj_error': projection_error_train,
                 'val_mse': val_loss,
                 'val_proj_error': projection_error_val,
                 'ww00': W_outer_pred[0, 0],
                 'ww01': W_outer_pred[0, 1],
                 'ww11': W_outer_pred[1, 1],
                 'w_outer_d': W_outer_d}
        nc_tracker.load_dt(nc_dt, epoch=epoch)

        wandb.log(
            {'train/lr': optimizer.param_groups[0]['lr'],
             'train/train_mse': train_loss,
             'train/project_error': projection_error_train,
             'val/val_mse': val_loss,
             'val/project_error': projection_error_val,
             'W/ww00': nc_dt['ww00'],
             'W/ww01': nc_dt['ww01'],
             'W/ww11': nc_dt['ww11'],
             'W/W_outer_d': W_outer_d,
             },
            step=epoch)


        log('Epoch {}/{}, runnning train mse: {:.4f}, ww00: {:.4f}, ww01: {:.4f}, ww11: {:.4f}, W_outer_d: {:.4f}'.format(
            epoch, args.max_epoch, train_loss, nc_dt['ww00'], nc_dt['ww01'], nc_dt['ww11'], W_outer_d
        ))

        if epoch % args.save_freq == 0 and args.dataset == 'Carla':
            ckpt_path = os.path.join(args.save_dir, 'ep{}_ckpt.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)

            log('--save model to {}'.format(ckpt_path))
        
        if epoch % (args.save_freq*10) == 0:
            filename = os.path.join(args.save_dir, 'train_nc{}.pkl'.format(epoch))
            with open(filename, 'wb') as f:
                pickle.dump(nc_tracker, f)


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
