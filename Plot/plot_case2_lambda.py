
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Plot.utils import add_headers

data = 'UTKFace'
path = f'result/rebuttal_UFM/{data}'

col_names = ['trainError', 'NRC1', 'NRC2', 'NRC3'] if data == 'Carla2D' else ['trainError', 'NRC1', 'NRC2']

result = []
for wd in ['0', '1e-2', '5e-3', '5e-4', '5e-5']:
    df_0 = pd.read_csv(os.path.join(path, f"WD{wd}/s0.csv"))
    df_0.columns = [name+f"_wd{wd}_sd0" for name in df_0.columns]

    df_2 = pd.read_csv(os.path.join(path, f"WD{wd}/s2.csv"))
    df_2.columns = [name + f"_wd{wd}_sd1" for name in df_2.columns]
    result += [df_0, df_2]

df = pd.concat(result, axis=1)

for wd in ['0', '1e-2', '5e-3', '5e-4', '5e-5']:
    for col_name in col_names:
        df[f'{col_name}_wd{wd}'] = (df[f'{col_name}_wd{wd}_sd0'] + df[f'{col_name}_wd{wd}_sd1'])/2
        df[f'{col_name}_wd{wd}'] = (df[f'{col_name}_wd{wd}_sd0'] + df[f'{col_name}_wd{wd}_sd1']) / 2


df1 = df


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
    ["C0", "C1", "C2", "C3", ],
    ["D0", "D1", "D2", "D3", ],
    ["E0", "E1", "E2", "." ],
    ["F0", "F1", "F2",  "." ],
]
row_headers = ["Reacher", "Swimmer", "Hopper", "CARLA2D", "CARLA1D", 'UTKFace']
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
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.01 - train/train_mse'].values, label='$\lambda_H=\lambda_W=$' +'1e-2', color=cmap1(norm(1)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=$' +'1e-3', color=cmap1(norm(2)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.0001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=$' +'1e-4', color=cmap1(norm(3)))
    axes[i].plot(acc['Step'].values*2, acc['wd: 0.00001 - train/train_mse'].values, label='$\lambda_H=\lambda_W=$' +'1e-5', color=cmap1(norm(4)))
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
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.01 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=$' +'1e-2', color=cmap1(norm(1)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=$' +'1e-3', color=cmap1(norm(2)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.0001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=$' +'1e-4', color=cmap1(norm(3)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0.00001 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=$' +'1e-5', color=cmap1(norm(4)))
    axes[i].plot(nc1['Step'].values*2, nc1['wd: 0 - train/train_nc1'].values, label='$\lambda_H=\lambda_W=0$', color=cmap1(norm(5)))
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.01 - train/train_nc1__MIN'].values, nc1['wd: 0.01 - train/train_nc1__MAX'].values, color=cmap1(norm(1)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.001 - train/train_nc1__MIN'].values, nc1['wd: 0.001 - train/train_nc1__MAX'].values, color=cmap1(norm(2)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.0001 - train/train_nc1__MIN'].values, nc1['wd: 0.0001 - train/train_nc1__MAX'].values, color=cmap1(norm(3)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0.00001 - train/train_nc1__MIN'].values, nc1['wd: 0.00001 - train/train_nc1__MAX'].values, color=cmap1(norm(4)), alpha=0.4)
    axes[i].fill_between(nc1['Step'].values*2, nc1['wd: 0 - train/train_nc1__MIN'].values, nc1['wd: 0 - train/train_nc1__MAX'].values, color=cmap1(norm(5)), alpha=0.4)
    axes[i].set_ylabel('NRC1')
    #axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')

    # ===== NC2
    i = row + '2'
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.01 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=$' +'1e-2', color=cmap1(norm(1)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=$' +'1e-3', color=cmap1(norm(2)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.0001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=$' +'1e-4', color=cmap1(norm(3)))
    axes[i].plot(nc3['Step'].values*2, nc3['wd: 0.00001 - train/train_nc3'].values, label='$\lambda_H=\lambda_W=$' +'1e-5', color=cmap1(norm(4)))
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
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.01 - W/nc2'].values, label='$\lambda_H=\lambda_W=$' +'1e-2', color=cmap1(norm(1)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.001 - W/nc2'].values, label='$\lambda_H=\lambda_W=$' +'1e-3', color=cmap1(norm(2)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.0001 - W/nc2'].values, label='$\lambda_H=\lambda_W=$' +'1e-4', color=cmap1(norm(3)))
    axes[i].plot(nc2['Step'].values*2, nc2['wd: 0.00001 - W/nc2'].values, label='$\lambda_H=\lambda_W=$' +'1e-5', color=cmap1(norm(4)))
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


# ========= For Carla
for idx, df in enumerate([df2, df1, df0]):
    row = ['D', 'E', 'F'][idx]
    i = row + '0'
    wd_list = ['1e-2', '5e-3', '5e-4', '5e-5', '0'] if idx==0 or idx==2 else ['1e-2', '5e-3', '5e-4', '0']
    if idx==2:
        steps = np.arange(51) * 2
    elif idx==1:
        steps = np.arange(100)
    elif idx==0:
        steps = np.arange(70)
        steps = np.round(steps * 100 / 70)

    # ==== Training Acc
    for j, wd in enumerate(wd_list):
        axes[i].plot(steps, df[f'trainError_wd{wd}'].values, label='$\lambda_H=\lambda_W=$'+wd, color=cmap1(norm(j+1)))
        axes[i].fill_between(steps,
                             np.minimum(df[f'trainError_wd{wd}_sd0'].values, df[f'trainError_wd{wd}_sd1'].values),
                             np.maximum(df[f'trainError_wd{wd}_sd0'].values, df[f'trainError_wd{wd}_sd1'].values), color=cmap1(norm(j+1)), alpha=0.4)
    axes[i].set_ylabel('Train MSE')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    if row =='E':
        offset_text = axes[i].yaxis.get_offset_text()
        offset_text.set_position((offset_text.get_position()[0]-0.05, -0.2))
    # axes[i].set_yscale("log")

    i = row + '1'
    # ===== NC1
    for j, wd in enumerate(wd_list):
        axes[i].plot(steps, df[f'NRC1_wd{wd}'].values, label='$\lambda_H=\lambda_W=$'+wd, color=cmap1(norm(j+1)))
        axes[i].fill_between(steps,
                             np.minimum(df[f'NRC1_wd{wd}_sd0'].values, df[f'NRC1_wd{wd}_sd1'].values),
                             np.maximum(df[f'NRC1_wd{wd}_sd0'].values, df[f'NRC1_wd{wd}_sd1'].values), color=cmap1(norm(j+1)), alpha=0.4)
    axes[i].set_ylabel('NRC1')
    # axes[i].set_yscale("log")
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')

    # ===== NC2
    i = row + '2'
    for j, wd in enumerate(wd_list):
        axes[i].plot(steps, df[f'NRC2_wd{wd}'].values, label='$\lambda_H=\lambda_W=$'+wd, color=cmap1(norm(j+1)))
        axes[i].fill_between(steps,
                             np.minimum(df[f'NRC2_wd{wd}_sd0'].values, df[f'NRC2_wd{wd}_sd1'].values),
                             np.maximum(df[f'NRC2_wd{wd}_sd0'].values, df[f'NRC2_wd{wd}_sd1'].values), color=cmap1(norm(j+1)), alpha=0.4)
    axes[i].set_ylabel('NRC2')
    axes[i].set_xlabel('Epoch')
    axes[i].grid(True, linestyle='--')
    # axes[i].set_yscale("log")
    # axes[i].set_xlim(0, 800)
    # axes[i].set_xticks([0, 200, 400, 600, 800])


    if idx == 0:
        i = row + '3'
        # ===== NC3
        for j, wd in enumerate(wd_list):
            axes[i].plot(steps, df[f'NRC3_wd{wd}'].values, label='$\lambda_H=\lambda_W=$'+wd, color=cmap1(norm(j+1)))
            axes[i].fill_between(steps,
                                 np.minimum(df[f'NRC3_wd{wd}_sd0'].values, df[f'NRC3_wd{wd}_sd1'].values),
                                 np.maximum(df[f'NRC3_wd{wd}_sd0'].values, df[f'NRC3_wd{wd}_sd1'].values), color=cmap1(norm(j+1)), alpha=0.4)

        axes[i].set_ylabel('NRC3')
        axes[i].set_xlabel('Epoch')
        axes[i].grid(True, linestyle='--')
        # axes[i].set_xlim(0, 800)
        # axes[i].set_xticks([0, 200, 400, 600, 800])

    axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Add a legend to the right-hand side for each row
handles, labels = [], []
for key, ax in axes.items():
    if key != ".":  # Skip empty subplots
        handle, label = ax.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)
        ax.legend().remove()

row_legend_y = np.arange(1-0.17, -0.6, -1/6) # Vertical position for each row legend
for i, row in enumerate(["A", "B", "C", "D", "E", "F"]):
    handles, labels = [], []
    if row in ['A', 'B', 'C', 'D']:
        key = row + '3'
    else:
        key = row + '2'
    handle, label = axes[key].get_legend_handles_labels()
    handles.extend(handle)
    labels.extend(label)
    axes[key].legend().remove()
    legend_ax = fig.add_axes([0.86, row_legend_y[i] - 0., 0.1, 0.2])  # Adjust these values as needed
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center')


