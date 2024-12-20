import torch
import numpy as np
import os

def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

def matrix_with_angle(angle=np.pi/4, dim=256):
    vec1 = np.random.randn(dim)
    vec1 /= np.linalg.norm(vec1)  # Normalize to make it a unit vector

    random_vec = np.random.randn(dim)
    orthogonal_vec = random_vec - vec1 * np.dot(random_vec, vec1)
    orthogonal_vec /= np.linalg.norm(orthogonal_vec)  # Normalize to make it a unit vector

    vec2 = np.cos(angle) * vec1 + np.sin(angle) * orthogonal_vec
    return np.concatenate((vec1.reshape(1, -1), vec2.reshape(1, -1)), axis=0)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count