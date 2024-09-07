import torch
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def get_scheduler(args, optimizer):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
        
    if args.scheduler in ['ms', 'multi_step']:
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epoch//4, args.max_epoch//2], gamma=0.2)
    elif args.scheduler in ['cos', 'cosine']:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


def get_feat_pred(model, loader):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            if target.ndim == 1: 
                target = target.unsqueeze(1)
            pred, feat = model(input, ret_feat=True)

            all_feats.append(feat)
            all_preds.append(pred)
            all_labels.append(target)

        all_feats = torch.cat(all_feats)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels).float()
    return all_feats, all_preds, all_labels


def get_all_feat(model, loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch_id, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)
            if target.ndim == 1:
                target = target.unsqueeze(1)
            feat_list = model.forward_feat(input)
            feat_list = [input] + feat_list

            if batch_id == 0:
                feat_by_layer = {layer_id: [] for layer_id in range(len(feat_list))}

            for layer_id, feat in enumerate(feat_list):
                feat_by_layer[layer_id].append(feat)

        for layer_id, layer_feat in feat_by_layer.items():
            feat_by_layer[layer_id] = torch.cat(layer_feat, dim=0)

    return feat_by_layer


def plot_var_ratio(vr_dt):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vr_dt.keys()), max(vr_dt.keys()))

    for id, vr_ratio in vr_dt.items():
        axes.plot(np.arange(len(vr_ratio)), vr_ratio, color=cmap(norm(id)), label=f'layer{id}')
    axes.legend()
    return fig


def plot_var_ratio_hw(vr_dt, wr_dt):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(vr_dt.keys()), max(vr_dt.keys()))

    for id, vr_ratio in vr_dt.items():
        wr_ratio = wr_dt[id]
        axes[0].plot(np.arange(len(vr_ratio)), vr_ratio, color=cmap(norm(id)), label=f'layer{id}')
        axes[1].plot(np.arange(len(wr_ratio)), wr_ratio, color=cmap(norm(id)), label=f'layer{id}')
    axes[0].legend()
    axes[1].legend()
    return fig


def compute_cosine_norm(W):
    # Compute cosine similarity between row vectors

    dot_product = torch.matmul(W, W.transpose(1, 0))  # Shape: (3, 3)

    # Compute L2 norm of row vectors
    norm = torch.sqrt(torch.sum(W ** 2, dim=-1, keepdim=True))  # Shape: (3, 1)

    # Compute cosine similarity
    cosine = dot_product / (norm * norm.transpose(1, 0))  # [3, 3]

    return cosine, norm


def gram_schmidt(W):
    dim = W.shape[0]

    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    for i in range(1, dim):
        j = i - 1
        ortho_vector = W[i, :]
        while j >= 0:
            proj = torch.dot(U[j, :], W[i, :]) * U[j, :]
            ortho_vector -= proj
            j -= 1
        U[i, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


# ============== NC metrics ==============
def compute_metrics(W, H, y_dim=None):
    result = {}
    H_row_norms = torch.norm(H, dim=1, keepdim=True)
    H_normalized = H / (H_row_norms + 1e-8)

    if y_dim == None:
        y_dim = W.shape[0]

    # NC1
    H_np = H.cpu().numpy()
    pca_for_H = PCA(n_components=31)
    try:
        pca_for_H.fit(H_np)
    except Exception as e:
        print(e)

    H_pca = torch.tensor(pca_for_H.components_[:max(y_dim, 6), :], device=H.device)  # First two principal components [2,5]
    H_U = gram_schmidt(H_pca)
    for k in range(max(y_dim, 6)):
        P_H = H_U[:k + 1, :].T @ H_U[:k + 1, :]
        result[f'nc1_pc{k + 1}'] = torch.sum((H @ P_H - H) ** 2).item() / len(H)
        result[f'nc1n_pc{k + 1}'] = torch.mean(torch.norm(H_normalized @ P_H - H_normalized, p=2, dim=1) ** 2).item()
        result[f'EVR{k + 1}'] = pca_for_H.explained_variance_ratio_[k]

    result['nc1'] = result[f'nc1_pc{y_dim}']
    result['nc1n'] = result[f'nc1n_pc{y_dim}']
    
    for k in range(5, 31, 5): 
        result[f'CVR{k}'] = np.sum(pca_for_H.explained_variance_ratio_[:k])
        
    try:
        inverse_mat = torch.inverse(W @ W.T)
    except Exception as e:
        print(e)
    P_W = W.T @ inverse_mat @ W
    result['nc2'] = torch.sum((H - H @ P_W)**2).item() / len(H)
    result['nc2n'] = torch.mean(torch.norm(H_normalized @ P_W - H_normalized, p=2, dim=1) ** 2).item()

    # Projection error with Gram-Schmidt
    # U = gram_schmidt(W)
    # P_E = torch.mm(U.T, U)
    # H_proj = torch.mm(H, P_E)
    # result['nc3a'] = torch.sum((H_proj-H)**2).item() / len(H)
    # del H_proj

    return result


def get_theoretical_solution(train_loader, args, all_labels=None, center=False):
    # if label not given, get all target label from data loader.
    if all_labels is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_labels = []
        with torch.no_grad():
            for i, (input, target) in enumerate(train_loader):
                target = target.to(device)
                all_labels.append(target)
            all_labels = torch.cat(all_labels).float()   # [N, 2]

    mu = torch.mean(all_labels, dim=0)
    if center:
        all_labels = all_labels - mu
    Sigma = (all_labels.T @ all_labels)/len(all_labels)
    Sigma = Sigma.cpu().numpy()

    if args.num_y >= 2:
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        Sigma_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)
        min_eigval = eigenvalues[0]
        max_eigval = eigenvalues[-1]
    elif args.num_y == 1: 
        Sigma_sqrt = np.sqrt(Sigma)
        min_eigval, max_eigval = Sigma_sqrt, Sigma_sqrt

    theory_stat = {
            'mu': mu.cpu().numpy(), 
            'Sigma': Sigma, 
            'Sigma_sqrt': Sigma_sqrt,
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
        }
    return theory_stat