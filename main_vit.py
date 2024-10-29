# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

import wandb
import timm
from models import *
from models.mlpmixer import MLPMixer
from models.convmixer import ConvMixer
from models.simplevit import SimpleViT
from models.vit import ViT
from models.vit_small import ViT as Small_ViT

from dataset import get_dataloader
from trainer import train_one_epoch, evaluate

from utils import *
from train_utils import get_feat_pred, analysis_feat, get_all_feat


def main(args):

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "./wandb"
    os.environ["WANDB_CONFIG_DIR"] = "./wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='NRC_sep',
               name=args.exp_name
               )
    wandb.config.update(args)

    global best_acc
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # ==================== data loader ====================
    print('==> Preparing data..')
    train_loader, test_loader = get_dataloader(args)

    # ====================  define model ====================
    # Model factory..
    print('==> Building model..')
    if args.model == 'res18' or args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'res50' or args.model == 'resnet50':
        model = ResNet50()
    elif args.model == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        model = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.model == "mlpmixer":
        model = MLPMixer(image_size=32, channels=3, patch_size=args.patch, dim=512, depth=6, num_classes=args.num_classes)
    elif args.model == "vit_small":
        model = Small_ViT(
            image_size=args.img_size, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=4, heads=6, mlp_dim=256,
            dropout=0.1, emb_dropout=0.1
        )
    elif args.model == "simplevit":
        model = SimpleViT(
            image_size=args.img_size, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=6, heads=8, mlp_dim=512
        )
    elif args.model == "vit":
        # ViT for cifar10
        model = ViT(
            image_size=args.img_size, patch_size=args.patch,
            num_classes=args.num_classes, dim=int(args.dimhead),
            depth=6, heads=8, mlp_dim=512,
            dropout=0.1, emb_dropout=0.1
        )
    elif args.model == "vit_timm":
        model = timm.create_model("vit_base_patch16_384", pretrained=True)
        model.head = nn.Linear(model.head.in_features, 10)
    elif args.model == "swin":
        from models.swin import swin_t
        model = swin_t(window_size=args.patch, num_classes=10, downscaling_factors=(2, 2, 2, 1))
    if torch.cuda.is_available():
        model.cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.model))
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # ====================  Training utilities ====================
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler in ['ms', 'multi_step']:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.max_epochs*0.3), int(args.max_epochs*0.6)], gamma=0.1)
    elif args.scheduler in ['cos', 'cosine']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    wandb.watch(model, criterion, log='all', log_freq=100)
    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, args)
        lr_scheduler.step()
        wandb.log({'train/train_loss': train_loss, 'train/train_acc': train_acc, 'train/lr': optimizer.param_groups[0]["lr"]}, step=epoch)

        if epoch == 0 or epoch % args.log_freq == 0 or epoch+1==args.max_epochs:
            val_loss, val_acc = evaluate(model, criterion, test_loader)

            all_feats, preds, labels = get_feat_pred(model, train_loader)
            labels = labels.squeeze()
            train_nc = analysis_feat(labels, all_feats, num_classes=args.num_classes, W=model.fc.weight.data)

            log_dt = {'val/val_loss': val_loss,
                      'val/val_acc': val_acc,

                      'train_nc/nc1': train_nc['nc1'],
                      'train_nc/nc2h': train_nc['nc2h'],
                      'train_nc/nc2w': train_nc['nc2w'],
                      'train_nc/nc2': train_nc['nc2'],
                      'train_nc/nc3': train_nc['nc3'],
                      'train_nc/h_norm': train_nc['h_norm'],
                      'train_nc/w_norm': train_nc['w_norm'],
                      }
            wandb.log(log_dt, step=epoch)

            weight_kqv = {id: model.transformer.layers[id][0].fn.to_qkv.weight for id in range(len(model.transformer.layers))}
            weight_att_out = {id: model.transformer.layers[id][0].fn.to_out[0].weight for id in range(len(model.transformer.layers))}
            weight_ffn = {id: model.transformer.layers[id][1].fn.net[0].weight for id in range(len(model.transformer.layers))}

            _, rk_weight_kqv = get_rank(weight_kqv)
            _, rk_weight_kqv = get_rank(weight_ffn)

            feat_by_layer = get_all_feat(model, train_loader, include_input=False, img_rs=False)
            _, rk_feat = get_rank(feat_by_layer)

            wandb.log({f'feat_rank/{id}': rk for id, rk in rk_feat.items()}, step=epoch)
            wandb.log({f'weight_kqv_rank/{id}': rk for id, rk in weight_kqv.items()}, step=epoch)
            wandb.log({f'weight_ffn_rank/{id}': rk for id, rk in weight_ffn.items()}, step=epoch)

        # ===== save model
        if args.save_ckpt and val_acc > best_acc:
            state = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),  # "scaler": scaler.state_dict()
                     }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + args.model + '-{}-ckpt.t7'.format(args.patch))
            best_acc = val_acc

        log('Epoch:{}, lr:{:.6f}, train loss:{:.4f}, train acc:{:.4f}; val loss:{:.4f}, val acc:{:.4f}'.format(
            epoch, optimizer.param_groups[0]["lr"], train_loss, train_acc, val_loss, val_acc
        ))


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model', default='vit')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--log_freq', default=5, type=int)
    parser.add_argument('--save_ckpt', default=False, action='store_true', help='save best model')

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_amp', default=False, action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--scheduler', type=str, default='ms')
    parser.add_argument('--max_epochs', type=int, default='200')

    # args for ViT
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

    parser.add_argument('--exp_name', type=str, default='baseline')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.img_size = 32
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.img_size = 32
        args.num_classes = 100

    args.output_dir = os.path.join('./result/{}_{}/'.format(args.dataset, args.model), args.exp_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    set_seed(SEED=args.seed)
    main(args)

