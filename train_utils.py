import torch
import os


def get_feat_pred(model, loader):
    model.eval()
    all_feats, all_preds, all_labels = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input = batch['image'].to(device)
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


def compute_cosine(W):
    # Compute cosine similarity between row vectors

    dot_product = torch.matmul(W, W.transpose(1, 0))  # Shape: (3, 3)

    # Compute L2 norm of row vectors
    norm = torch.sqrt(torch.sum(W ** 2, dim=-1, keepdim=True))  # Shape: (3, 1)

    # Compute cosine similarity
    cosine = dot_product / (norm * norm.transpose(1, 0))

    return cosine


def gram_schmidt(W):
    U = torch.empty_like(W)
    U[0, :] = W[0, :] / torch.norm(W[0, :], p=2)

    proj = torch.dot(U[0, :], W[1, :]) * U[0, :]
    ortho_vector = W[1, :] - proj
    U[1, :] = ortho_vector / torch.norm(ortho_vector, p=2)

    return U


class Graph_Vars:
    def __init__(self):
        self.epoch = []
        self.lr = []
        
        self.train_mse = []
        self.val_mse = []

        self.train_proj_error = []
        self.val_proj_error = []
        self.cos_w12 = []
        self.cos_w13 = []
        self.cos_w23 = []

    def load_dt(self, nc_dt, epoch, lr=None):
        self.epoch.append(epoch)
        if lr:
            self.lr.append(lr)
        for key in nc_dt:
            try:
                self.__getattribute__(key).append(nc_dt[key])
            except:
                print('{} is not attribute of Graph var'.format(key))