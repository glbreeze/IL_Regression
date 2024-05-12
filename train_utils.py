import torch
import os
import numpy as np
import torch.optim as optim


def get_scheduler(args, optimizer):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """

    SCHEDULERS = {
        'multi_step': optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300], gamma=0.2),
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch),
    }
    return SCHEDULERS[args.scheduler]


def get_feat_pred(model, loader):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)
            input, target = input.to(device), target.to(device)
            pred, feat = model(input, ret_feat=True)

            all_feats.append(feat)
            all_preds.append(pred)
            all_labels.append(target)

        all_feats = torch.cat(all_feats)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
    return all_feats, all_preds, all_labels


def compute_cosine_norm(W):
    # Compute cosine similarity between row vectors

    dot_product = torch.matmul(W, W.transpose(1, 0))  # Shape: (3, 3)

    # Compute L2 norm of row vectors
    norm = torch.sqrt(torch.sum(W ** 2, dim=-1, keepdim=True))  # Shape: (3, 1)

    # Compute cosine similarity
    cosine = dot_product / (norm * norm.transpose(1, 0))  # [3, 3]

    return cosine, norm


def gram_schmidt(W):
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


class Graph_Vars:
    def __init__(self, dim=2):
        self.epoch = []
        self.lr = []

        self.train_mse = []
        self.train_proj_error = []
        self.val_mse = []
        self.val_proj_error = []
        self.w_outer_d = []
        self.ww00 = []
        if dim==2:
            self.ww01 = []
            self.ww11 = []

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


class Train_Vars:
    def __init__(self, dim=2):
        self.epoch = []
        self.lr = []

        self.train_mse = []
        self.train_proj_error = []
        self.w_outer_d = []
        self.ww00 = []
        if dim==2:
            self.ww01 = []
            self.ww11 = []

    def load_dt(self, nc_dt, epoch):
        self.epoch.append(epoch)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))


def get_theoretical_solution(train_loader, args, bias=None, all_labels=None, center=False):
    if all_labels is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                target = batch['target'].to(device)
                all_labels.append(target)
            all_labels = torch.cat(all_labels)   # [N, 2]

    if bias is not None:
        center_labels = all_labels - bias  # [N,2] - [2]
    else:
        center_labels = all_labels
    if center:
        mu = torch.mean(all_labels, dim=0)
        center_labels = all_labels - mu
    else:
        mu = torch.mean(all_labels, dim=0)
        center_labels = all_labels
    Sigma = torch.matmul(center_labels.T, center_labels)/len(center_labels)
    Sigma = Sigma.cpu().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    Sigma_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
    min_eigval = eigenvalues[0]
    max_eigval = eigenvalues[-1]
    
    mu11 = Sigma[0, 0]
    mu12 = Sigma[0, 1]
    mu22 = Sigma[1, 1]

    W_outer = args.lambda_H * (Sigma_sqrt/np.sqrt(args.lambda_H*args.lambda_W) - np.eye(args.num_y))
    theory_stat = {
            'mu11': Sigma[0, 0],
            'mu12': Sigma[0, 1],
            'mu22': Sigma[1, 1],
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
            'sigma11': Sigma_sqrt[0, 0],
            'sigma12': Sigma_sqrt[0, 1],
            'sigma21': Sigma_sqrt[1, 0],
            'sigma22': Sigma_sqrt[1, 1],
            'mu1': mu[0].item(),
            'mu2': mu[1].item()
        }
    return W_outer, Sigma_sqrt, all_labels, theory_stat # all_labels is still tensor    