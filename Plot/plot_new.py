


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers

data = 'carla1d'
exp = 'ufm'
seed = 2023
folder = f'result/{data}/{exp}/seed{seed}'

if exp == 'case1':
    file = f'{data}_s{seed}_mse_train.csv'
else:
    file = f'{data}_ufm_s{seed}_mse_val.csv'

df = pd.read_csv(os.path.join(folder, file))
df.columns = df.columns.str.replace('_LR1e-3_s2023', '')
df.to_csv(os.path.join(folder, file))



# ============================ colar mapping ============================
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 4)

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ]
]
row_headers = ["Case1", "Case2"]
col_headers = ["Train MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

data = 'carla'

for row in ['A','B']:
    if row == 'A':
        exp = 'case1'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_s{2023}'
    elif row == 'B':
        exp = 'ufm'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_ufm_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_ufm_s{2023}'

    nc1 = pd.read_csv(folder1 + '_nc1n.csv')
    nc2 = pd.read_csv(folder1 + '_nc2n.csv')
    nc3 = pd.read_csv(folder1 + '_nc3.csv')
    mse_train = pd.read_csv(folder1 + '_mse_train.csv')
    mse_val = pd.read_csv(folder1 + '_mse_val.csv')

    i = row + '0'
    # ==== Training Acc
    for j, wd in enumerate(['1e-1', '5e-2', '1e-2', '5e-3', '1e-3']):
        col_name = f'{data}_res18_WD{wd} - mse/train_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/train_mse'
        axes[i].plot(nc1['Step'].values, mse_train[col_name].values, label='$\lambda_H=\lambda_W=$'+'wd', color=cmap1(norm(j)), linestyle='-')
    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    print('------')

    i = row + '1'
    # ==== Training Acc
    for j, wd in enumerate(['1e-1', '5e-2', '1e-2', '5e-3', '1e-3']):
        col_name = f'{data}_res18_WD{wd} - mse/val_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/val_mse'
        axes[i].plot(nc1['Step'].values, mse_val[col_name].values, label='$\lambda_H=\lambda_W=$' + 'wd', color=cmap1(norm(j)), linestyle='-')
    axes[i].set_ylabel('Val MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")



# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ]
]
row_headers = ["Case1", "Case2"]
col_headers = ["Train MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

data = 'carla'

for row in ['A','B']:
    if row == 'A':
        exp = 'case1'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_s{2023}'
    elif row == 'B':
        exp = 'ufm'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_ufm_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_ufm_s{2023}'

    nc1 = pd.read_csv(folder1 + '_nc1n.csv')
    nc2 = pd.read_csv(folder1 + '_nc2n.csv')
    nc3 = pd.read_csv(folder1 + '_nc3.csv')
    mse_train = pd.read_csv(folder1 + '_mse_train.csv')
    mse_val = pd.read_csv(folder1 + '_mse_val.csv')

    nc1_2 = pd.read_csv(folder2 + '_nc1n.csv')
    nc2_2 = pd.read_csv(folder2 + '_nc2n.csv')
    nc3_2 = pd.read_csv(folder2 + '_nc3.csv')
    mse_train_2 = pd.read_csv(folder2 + '_mse_train.csv')
    mse_val_2 = pd.read_csv(folder2 + '_mse_val.csv')

    i = row + '0'
    # ==== Training Acc
    j, wd = 1, '5e-2'
    col_name = f'{data}_res18_WD{wd} - mse/train_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/train_mse'
    axes[i].plot(nc1['Step'].values, mse_train[col_name].values, label='$\lambda_H=\lambda_W=$'+'wd', color=cmap1(norm(j)), linestyle='-')
    axes[i].fill_between(nc1['Step'].values, mse_train[col_name].values, mse_train_2[col_name].values, color=cmap1(norm(j)), alpha=0.4)

    col_name = f'{data}_res18_WD{wd} - mse/val_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/val_mse'
    axes[i].plot(nc1['Step'].values, mse_val[col_name].values, label='$\lambda_H=\lambda_W=$' + 'wd', color=cmap1(norm(j+2)), linestyle='--')


    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    print('------')





# ============================ colar mapping ============================
# Normalize x values to [0, 1] for colormap mapping
cmap1 = plt.cm.viridis
cmap2 = plt.cm.plasma
norm = plt.Normalize(1, 4)

# ============================ Plot ============================
mosaic = [
    ["A0", "A1", "A2", "A3", ],
    ["B0", "B1", "B2", "B3", ]
]
row_headers = ["Case1", "Case2"]
col_headers = ["Train MSE", "NRC1", "NRC2", "NRC3"]

subplots_kwargs = dict(sharex=False, sharey=False, figsize=(10, 6))
fig, axes = plt.subplot_mosaic(mosaic, **subplots_kwargs)

font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize="large")
add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

data = 'carla'

for row in ['A','B']:
    if row == 'A':
        exp = 'case1'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_s{2023}'
    elif row == 'B':
        exp = 'ufm'
        folder1 = f'result/{data}/{exp}/seed2021/{data}_ufm_s{2021}'
        folder2 = f'result/{data}/{exp}/seed2023/{data}_ufm_s{2023}'

    nc1 = pd.read_csv(folder1 + '_nc1n.csv')
    nc2 = pd.read_csv(folder1 + '_nc2n.csv')
    nc3 = pd.read_csv(folder1 + '_nc3.csv')
    mse_train = pd.read_csv(folder1 + '_mse_train.csv')
    mse_val = pd.read_csv(folder1 + '_mse_val.csv')

    nc1_2 = pd.read_csv(folder2 + '_nc1n.csv')
    nc2_2 = pd.read_csv(folder2 + '_nc2n.csv')
    nc3_2 = pd.read_csv(folder2 + '_nc3.csv')
    mse_train_2 = pd.read_csv(folder2 + '_mse_train.csv')
    mse_val_2 = pd.read_csv(folder2 + '_mse_val.csv')

    i = row + '0'
    # ==== Training Acc
    for j, wd in enumerate([ '5e-2', '1e-2', '5e-3', '1e-3']):
        col_name = f'{data}_res18_WD{wd} - mse/train_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/train_mse'
        axes[i].plot(nc1['Step'].values, mse_train[col_name].values, label='$\lambda_H=\lambda_W=$'+'wd', color=cmap1(norm(j)), linestyle='-')
        axes[i].fill_between(nc1['Step'].values, mse_train[col_name].values, mse_train_2[col_name].values, color=cmap1(norm(j)), alpha=0.4)
    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")

    print('------')

    i = row + '1'
    # ==== Training Acc
    for j, wd in enumerate([ '5e-2', '1e-2', '5e-3', '1e-3']):
        col_name = f'{data}_res18_WD{wd} - mse/val_mse' if exp == 'case1' else f'{data}_ufm_res18_WD{wd} - mse/val_mse'
        axes[i].plot(nc1['Step'].values, mse_val[col_name].values, label='$\lambda_H=\lambda_W=$' + 'wd', color=cmap1(norm(j)), linestyle='-')
        axes[i].fill_between(nc1['Step'].values, mse_val[col_name].values, mse_val_2[col_name].values, color=cmap1(norm(j)), alpha=0.4)
    axes[i].set_ylabel('Val MSE')
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
