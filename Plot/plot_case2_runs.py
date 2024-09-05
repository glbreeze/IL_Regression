

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers

mapping = {'A': 're', 'B':'sw', 'C':'hp'}

# ========= colar mapping =========
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 5)

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ],
    ["C0", "C1", "C2", "C3", ]
]
row_headers = ["Reacher", "Swimmer", "Hopper"]
col_headers = ["Train MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for row in ['A','B','C']:
    if row == 'A':
        path = '../result/reacher/ufm/'
    elif row == 'B':
        path = '../result/swimmer/ufm/'
    elif row == 'C':
        path = '../result/hopper/ufm/'

    nc1 = pd.read_csv(os.path.join(path, mapping[row]+'_nc1.csv')).head(500)
    nc2 = pd.read_csv(os.path.join(path, mapping[row]+'_nc2.csv')).head(500)
    nc3 = pd.read_csv(os.path.join(path, mapping[row]+'_nc3.csv')).head(500)
    acc = pd.read_csv(os.path.join(path, mapping[row]+'_mse.csv')).head(500)

    i = row + '0'
    # ==== Training Acc
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.01 - train/train_mse'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.0001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.00001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0 - train/train_mse'].values, label='$\lambda_H=\lambda_W=0$', color=cmap1(norm(5)))
    axes[i].fill_between(acc['Step'].values*2, acc['wd: 0.01 - train/train_mse__MIN'].values, acc['wd: 0.01 - train/train_mse__MAX'].values, color=cmap1(norm(1)), alpha=0.4)
    axes[i].fill_between(acc['Step'].values*2, acc['wd: 0.001 - train/train_mse__MIN'].values, acc['wd: 0.001 - train/train_mse__MAX'].values, color=cmap1(norm(2)), alpha=0.4)
    axes[i].fill_between(acc['Step'].values*2, acc['wd: 0.0001 - train/train_mse__MIN'].values, acc['wd: 0.0001 - train/train_mse__MAX'].values, color=cmap1(norm(3)), alpha=0.4)
    axes[i].fill_between(acc['Step'].values*2, acc['wd: 0.00001 - train/train_mse__MIN'].values, acc['wd: 0.00001 - train/train_mse__MAX'].values, color=cmap1(norm(4)), alpha=0.4)
    axes[i].fill_between(acc['Step'].values*2, acc['wd: 0 - train/train_mse__MIN'].values, acc['wd: 0 - train/train_mse__MAX'].values, color=cmap1(norm(5)), alpha=0.4)

    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    i = row + '1'
    # ===== NC1
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.01 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.0001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.00001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=0$', color=cmap1(norm(5)))
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.01 - train/train_nc1__MIN'].values, nc1['wd: 0.01 - train/train_nc1__MAX'].values, color=cmap1(norm(1)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.001 - train/train_nc1__MIN'].values, nc1['wd: 0.001 - train/train_nc1__MAX'].values, color=cmap1(norm(2)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.0001 - train/train_nc1__MIN'].values, nc1['wd: 0.0001 - train/train_nc1__MAX'].values, color=cmap1(norm(3)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.00001 - train/train_nc1__MIN'].values, nc1['wd: 0.00001 - train/train_nc1__MAX'].values, color=cmap1(norm(4)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0 - train/train_nc1__MIN'].values, nc1['wd: 0 - train/train_nc1__MAX'].values, color=cmap1(norm(5)), alpha=0.4)
    axes[i].set_ylabel('NRC1')
    axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')

    # ===== NC2
    i = row + '2'
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.01 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.0001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.00001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=0$', color=cmap1(norm(5)))
    axes[i].fill_between(nc3['Step'].values*2, nc3['wd: 0.01 - train/train_nc3__MIN'].values, nc3['wd: 0.01 - train/train_nc3__MAX'].values, color=cmap1(norm(1)), alpha=0.4)
    axes[i].fill_between(nc3['Step'].values*2, nc3['wd: 0.001 - train/train_nc3__MIN'].values, nc3['wd: 0.001 - train/train_nc3__MAX'].values, color=cmap1(norm(2)), alpha=0.4)
    axes[i].fill_between(nc3['Step'].values*2, nc3['wd: 0.0001 - train/train_nc3__MIN'].values, nc3['wd: 0.0001 - train/train_nc3__MAX'].values, color=cmap1(norm(3)), alpha=0.4)
    axes[i].fill_between(nc3['Step'].values*2, nc3['wd: 0.00001 - train/train_nc3__MIN'].values, nc3['wd: 0.00001 - train/train_nc3__MAX'].values, color=cmap1(norm(4)), alpha=0.4)
    axes[i].fill_between(nc3['Step'].values*2, nc3['wd: 0 - train/train_nc3__MIN'].values, nc3['wd: 0 - train/train_nc3__MAX'].values, color=cmap1(norm(5)), alpha=0.4)
    axes[i].set_ylabel('NRC2')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])



    i = row + '3'
    # ===== NC3
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.01 - W/nc2'].values, label='$\lambda_H=\lambda_W=1e-2$', color=cmap1(norm(1)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.001 - W/nc2'].values, label='$\lambda_H=\lambda_W=1e-3$', color=cmap1(norm(2)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.0001 - W/nc2'].values, label='$\lambda_H=\lambda_W=1e-4$', color=cmap1(norm(3)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.00001 - W/nc2'].values, label='$\lambda_H=\lambda_W=1e-5$', color=cmap1(norm(4)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0 - W/nc2'].values, label='$\lambda_H=\lambda_W=0$', color=cmap1(norm(5)))
    axes[i].fill_between(nc2['Step'].values*2, nc2['wd: 0.01 - W/nc2__MIN'].values, nc2['wd: 0.01 - W/nc2__MAX'].values, color=cmap1(norm(1)), alpha=0.4)
    axes[i].fill_between(nc2['Step'].values*2, nc2['wd: 0.001 - W/nc2__MIN'].values, nc2['wd: 0.001 - W/nc2__MAX'].values, color=cmap1(norm(2)), alpha=0.4)
    axes[i].fill_between(nc2['Step'].values*2, nc2['wd: 0.0001 - W/nc2__MIN'].values, nc2['wd: 0.0001 - W/nc2__MAX'].values, color=cmap1(norm(3)), alpha=0.4)
    axes[i].fill_between(nc2['Step'].values*2, nc2['wd: 0.00001 - W/nc2__MIN'].values, nc2['wd: 0.00001 - W/nc2__MAX'].values, color=cmap1(norm(4)), alpha=0.4)
    axes[i].fill_between(nc2['Step'].values*2, nc2['wd: 0 - W/nc2__MIN'].values, nc2['wd: 0 - W/nc2__MAX'].values, color=cmap1(norm(5)), alpha=0.4)

    axes[i].set_ylabel('NRC3')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))





    # i = row + '4'
    # # ==== Testing Acc
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0'].values, label='$\delta=0$', color=cmap1(norm(1)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.05'].values, label='$\delta=0.05$', color=cmap1(norm(2)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.1'].values, label='$\delta=0.1$', color=cmap1(norm(3)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.2'].values, label='$\delta=0.2$', color=cmap1(norm(4)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.3'].values, label='$\delta=0.3$', color=cmap1(norm(5)))
    # axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.4'].values, label='$\delta=0.4$', color=cmap1(norm(6)))
    # #axes[i].plot(acc['Step'].values, 1 - test_acc['ls0.5'].values, label='$\delta=0.5$', color=cmap1(norm(7)))
    # #axes[i].plot(np.arange(800), 1 - test_acc['ls0.95'].values, label='$\delta=0.95$', color=cmap1(norm(8)))
    # axes[i].set_ylabel('Test Error Rate')
    # axes[i].set_xlabel('Epoch')
    # axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    # axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #fig.suptitle("Model Convergence with Different Smoothing Hyperparameter")


