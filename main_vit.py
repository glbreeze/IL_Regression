# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import wandb
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.sheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.disc_utils import get_world_size
from utils.utils import AverageMeter, get_rank
from utils.train_utils import get_feat_pred, analysis_feat
from dataset import get_dataloader

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "best_ckpt.pth")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.amp))

    # Set seed
    set_seed(args)

    # setup wandb
    if args.local_rank in [-1, 0]:
        os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
        os.environ["WANDB_MODE"] = "online"  # "dryrun"
        os.environ["WANDB_CACHE_DIR"] = "./wandb"
        os.environ["WANDB_CONFIG_DIR"] = "./wandb"
        wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
        wandb.init(project='NRC_sep', name=args.exp_name)
        wandb.config.update(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model]
    num_classes = 100 if args.dataset == "im100" else 1000
    args.output_dir = os.path.join('./result/{}_{}/'.format(args.dataset, args.model), args.exp_name)
    if args.local_rank in [-1, 0]:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def train(args, model):
    """ Train the model """
    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    # ============ Prepare dataset  ============
    train_loader, test_loader = get_dataloader(args)
    args.num_steps = len(train_loader) * args.num_epochs / args.gradient_accumulation_steps

    #  ============ Prepare optimizer and scheduler  ============
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # ============= Train cls =============
    if args.cls_epochs>0:
        train_cls(args, model, train_loader, test_loader)

    #  ============ Train  ============
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.num_steps}, Total number of epochs = {args.num_epochs}", )
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    global_step, best_acc = 0, 0
    scaler = amp.GradScaler() if args.amp else None
    for epoch in range(args.num_epochs):
        model.train()
        losses = AverageMeter('train_loss')
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True, disable=args.local_rank not in [-1, 0])

        for step, (x, y) in enumerate(epoch_iterator):
            x, y = x.to(args.device), y.to(args.device)
            with amp.autocast(dtype=torch.float16, enabled=args.amp):
                loss = model(x, y)
                losses.update(loss.item())
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.amp:
                    scaler.scale(loss).backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                global_step += 1
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, losses.val)
                )
                if args.local_rank in [-1, 0] and global_step % args.log_every == 0:
                    wandb.log({"train/loss": losses.val,
                               "train/lr": scheduler.get_lr()[0]
                               }, step=global_step)
                    losses.reset()

                    weight_q = {id: model.transformer.encoder.layer[id].attn.query.weight.data.cpu().numpy() for id in
                                range(len(model.transformer.layers))}
                    _, rk_weight_q = get_rank(weight_q)
                    wandb.log({f'weight_q_rank/{id}': rk for id, rk in rk_weight_q.items()}, step=global_step)
                    del weight_q

                    weight_k = {id: model.transformer.encoder.layer[id].attn.key.weight.data.cpu().numpy() for id in
                                range(len(model.transformer.layers))}
                    _, rk_weight_k = get_rank(weight_k)
                    wandb.log({f'weight_k_rank/{id}': rk for id, rk in rk_weight_k.items()}, step=global_step)
                    del weight_k

                    weight_v = {id: model.transformer.encoder.layer[id].attn.value.weight.data.cpu().numpy() for id in
                                range(len(model.transformer.layers))}
                    _, rk_weight_v = get_rank(weight_v)
                    wandb.log({f'weight_v_rank/{id}': rk for id, rk in rk_weight_v.items()}, step=global_step)
                    del weight_v

                    weight_ffn = {id: model.transformer.encoder.layer[id].ffn.fc1.weight.data.cpu().numpy() for id in
                                  range(len(model.transformer.encoder.layer))}
                    _, rk_weight_ffn = get_rank(weight_ffn)
                    wandb.log({f'weight_ffn_rank/{id}': rk for id, rk in rk_weight_ffn.items()}, step=epoch)
                    del weight_ffn

                if args.local_rank in [-1, 0] and global_step % args.eval_every == 0:
                    # all_feats, preds, labels = get_feat_pred(model, train_loader)
                    # labels = labels.squeeze()
                    # train_nc = analysis_feat(labels, all_feats, num_classes=args.num_classes, W=model.fc.weight.data)

                    test_acc = valid(args, model, test_loader)
                    wandb.log({"val/acc": test_acc}, step=global_step)
                    if best_acc < test_acc:
                        save_model(args, model)
                        best_acc = test_acc
                    model.train()

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def train_cls(args, model, train_loader, test_loader):
    set_seed(args)
    for param in model.transformer.parameters():
        param.requires_grad = False
    cls_optimizer = torch.optim.SGD(model.head.parameters(), lr=args.cls_lr, momentum=0.9, weight_decay=args.weight_decay)

    for epoch in range(args.cls_epochs):
        model.zero_grad()
        model.transformer.eval()
        model.head.train()
        losses = AverageMeter('train_cls_loss')

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            loss = model(x, y)
            losses.update(loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.max_grad_norm)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

                prog = (epoch * len(train_loader) + step) / (args.cls_epochs * len(train_loader))
                cls_optimizer.param_groups[0]['lr'] = args.cls_lr * 0.2 ** (prog // 0.5)
        print(f'Train Classifier EP{epoch}/{args.cls_epochs} Train loss:{losses.avg:.3f}, LR:{cls_optimizer.param_groups[0]["lr"]:.5f}')

        test_acc = valid(args, model, test_loader)
        print(f'Train Classifier EP{epoch}/{args.cls_epochs} Test Acc:{test_acc:.3f}')

    for param in model.transformer.parameters():
        param.requires_grad = True


def valid(args, model, test_loader):
    eval_losses = AverageMeter('val_loss')

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, (x, y) in enumerate(epoch_iterator):
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--exp_name", default='exp', help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", "im100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model",
                        choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="models/pretrain/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--eval_every", default=100, type=int, help="Run prediction on validation set every so many steps.")
    parser.add_argument("--log_every", default=10, type=int, help="Log train loss every so many steps.")

    parser.add_argument("--lr", default=3e-2, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--scheduler", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--cls_epochs", default=0, type=int, help="Total number of epochs to retrain cls layer.")
    parser.add_argument("--cls_lr", default=1e-3, type=float, help="LR for training cls.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--amp', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()
    main(args)
