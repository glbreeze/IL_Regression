import os 
import shutil
import random
import pickle
import numpy as np 

folder = '/vast/lg154/Carla_JPG/Train'
files1 = os.listdir(os.path.join(folder, 'images'))
files2 = os.listdir(os.path.join(folder, 'targets'))

random.seed(2023)
selected = random.sample(files1, 50000)
selected.sort()

# store the images to a new folder 
new_folder = os.path.join(folder, 'sub_images')

all_targets = []
filename = os.path.join(folder, 'train_list.txt')
with open(filename, 'w') as f: 
    for item in selected: 
        shutil.copy(os.path.join(folder, 'images', item), new_folder)
        f.write(str(item) + '\n')
        
        idx = item.split('_')[1].split('.')[0]
        target_path = os.path.join(folder, 'targets', 'target_{}.npy'.format(idx))
        target = np.load(target_path)
        all_targets.append(target.reshape(1,-1))
        
all_targets = np.concatenate(all_targets, axis=0)
print('----target shape: {}'.format(all_targets.shape))

filename = os.path.join(folder, 'sub_targets.pkl')
with open(filename, 'wb') as file:
    pickle.dump(all_targets, file)
print('---has saved target to {}'.format(filename))
        


