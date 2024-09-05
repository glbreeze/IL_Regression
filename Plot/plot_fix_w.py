

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers

mapping = {'A': 're', 'B':'sw', 'C':'hp'}

# ========= colar mapping =========
# Normalize x values to [0, 1] for colormap mapping


# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", 'A4'],
    ["B0", "B1", "B2", "B3", 'B4'],
    ["C0", "C1", "C2", "C3", 'C4']
]
row_headers = ["Reacher", "Swimmer", "Hopper"]
col_headers = ["Train MSE", "Test MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)


for row in ['A','B','C']:
    if row == 'A':
        path = '../result/reacher/fix_w/'

    elif row == 'B':
        path = '../result/swimmer/fix_w/'

    elif row == 'C':
        path = '../result/hopper/fix_w/'


    nc1 = pd.read_csv(os.path.join(path, mapping[row]+'_nc1.csv'))
    nc2 = pd.read_csv(os.path.join(path, mapping[row]+'_nc2.csv'))
    nc3 = pd.read_csv(os.path.join(path, mapping[row]+'_nc3.csv'))
    acc = pd.read_csv(os.path.join(path, mapping[row]+'_mse_train.csv'))
    acc_test = pd.read_csv(os.path.join(path, mapping[row] + '_mse_test.csv'))

    if row == 'A':
        N = 1000
    elif row == 'B':
        N = 200
    elif row == 'C':
        N = 200
    nc1 = nc1.head(N)
    nc2 = nc2.head(N)
    nc3 = nc3.head(N)
    acc = acc.head(N)
    acc_test = acc_test.head(N)



    i = row + '0'
    # ==== Training Acc
    steps = acc['Step'].values
    axes[i].plot(steps, acc['Wn'].values, label='Baseline')
    axes[i].plot(steps, acc['We1'].values, label='fixed W')
    # axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].legend()
    if row == 'A':
        axes[i].set_xticks([0, 10000, 20000])
        axes[i].set_xticklabels(['0', '0.5M', '1M'])
    elif row == 'B':
        axes[i].set_xticks([0, 50000, 100000])
        axes[i].set_xticklabels(['0', '30K', '60K'])
    elif row == 'C':
        axes[i].set_xticks([0, 5000, 10000])
        axes[i].set_xticklabels(['0', '60K', '120K'])

    i = row + '1'
    # ==== Training Acc
    axes[i].plot(acc_test['Step'].values, acc_test['Wn'].values, label='Baseline')
    axes[i].plot(acc_test['Step'].values, acc_test['We1'].values, label='fixed W')
    # axes[i].set_ylabel('Test MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].legend()
    # axes[i].set_yscale("log")
    if row == 'A':
        axes[i].set_xticks([0, 10000, 20000])
        axes[i].set_xticklabels(['0', '0.5M', '1M'])
    elif row == 'B':
        axes[i].set_xticks([0, 50000, 100000])
        axes[i].set_xticklabels(['0', '30K', '60K'])
    elif row == 'C':
        axes[i].set_xticks([0, 5000, 10000])
        axes[i].set_xticklabels(['0', '60K', '120K'])

    i = row + '2'
    # ===== NC1
    axes[i].plot(nc1['Step'].values, nc1['Wn'].values, label='Baseline')
    axes[i].plot(nc1['Step'].values, nc1['We1'].values, label='fixed W')
    # axes[i].set_ylabel('NRC1')
    # axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].legend()
    if row == 'A':
        axes[i].set_xticks([0, 10000, 20000])
        axes[i].set_xticklabels(['0', '0.5M', '1M'])
    elif row == 'B':
        axes[i].set_xticks([0, 50000, 100000])
        axes[i].set_xticklabels(['0', '30K', '60K'])
    elif row == 'C':
        axes[i].set_xticks([0, 5000, 10000])
        axes[i].set_xticklabels(['0', '60K', '120K'])

    # ===== NC2
    i = row + '3'
    axes[i].plot(nc3['Step'].values, nc3['Wn'].values, label='Baseline')
    axes[i].plot(nc3['Step'].values, nc3['We1'].values, label='fixed W')
    # axes[i].set_ylabel('NRC2')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    axes[i].legend()
    if row == 'A':
        axes[i].set_xticks([0, 10000, 20000])
        axes[i].set_xticklabels(['0', '0.5M', '1M'])
    elif row == 'B':
        axes[i].set_xticks([0, 50000, 100000])
        axes[i].set_xticklabels(['0', '30K', '60K'])
    elif row == 'C':
        axes[i].set_xticks([0, 5000, 10000])
        axes[i].set_xticklabels(['0', '60K', '120K'])
    # axes[i].set_yscale("log")
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])



    i = row + '4'
    # ===== NC3
    axes[i].plot(nc2['Step'].values, nc2['Wn'].values, label='Baseline')
    axes[i].plot(nc2['Step'].values, nc2['We1'].values, label='fixed W')

    # axes[i].set_ylabel('NRC3')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])
    axes[i].legend()
    if row == 'A':
        axes[i].set_xticks([0, 10000, 20000])
        axes[i].set_xticklabels(['0', '0.5M', '1M'])
    elif row == 'B':
        axes[i].set_xticks([0, 50000, 100000])
        axes[i].set_xticklabels(['0', '30K', '60K'])
    elif row == 'C':
        axes[i].set_xticks([0, 5000, 10000])
        axes[i].set_xticklabels(['0', '60K', '120K'])




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


