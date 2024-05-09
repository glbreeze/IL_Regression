import os
import h5py
import pickle 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

DATA_FOLDER = '../dataset/mujoco_data/'


def get_dataloader(args):
    if args.dataset == 'Carla' or args.dataset == 'carla':
        train_dataset = SubDataset('/vast/lg154/Carla_JPG/Train/train_list.txt', '/vast/lg154/Carla_JPG/Train/sub_targets.pkl', transform=transform, dim=args.num_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = train_loader
    elif args.dataset == "swimmer" or args.dataset == 'reacher':
        if args.dataset == 'swimmer':
            DATA_RATIO = 0.1
        else:
            DATA_RATIO = 1.0
        train_dataset = MujocoBuffer(data_folder=DATA_FOLDER,
            env=args.dataset,
            split='train',
            data_ratio=DATA_RATIO,
            args=args
        )
        val_dataset = MujocoBuffer(
            data_folder=DATA_FOLDER,
            env=args.dataset,
            split='test',
            data_ratio=DATA_RATIO,
            args=args,
            y_shift=train_dataset.y_shift,
            div=train_dataset.div
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


class ImageTargetDataset(Dataset):

    def __init__(self, images_dir, targets_dir, transform=None, dim=2):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.dim = dim

        self.image_filenames = [f for f in sorted(os.listdir(images_dir)) if f.endswith('.jpeg')]
        self.target_filenames = [f for f in sorted(os.listdir(targets_dir)) if f.endswith('.npy')]

        if len(self.image_filenames) != len(self.target_filenames):
            raise ValueError("The number of images and targets do not match!")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        target_path = os.path.join(self.targets_dir, self.target_filenames[idx])

        image = Image.open(img_path).convert('RGB')
        target = np.load(target_path)

        if self.transform:
            image = self.transform(image)

        if self.dim==1:
            return {'input': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'input': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


class SubDataset(Dataset):

    def __init__(self, train_list, target_file, transform=None, dim=2):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        folder = os.path.dirname(train_list)
        self.img_path = []

        with open(train_list, 'r') as f:
            for line in f:
                self.img_path.append(os.path.join(folder, 'sub_images', line.strip()))
        
        with open(target_file, 'rb') as file:
            self.targets = pickle.load(file)
                
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):

        img_path = self.img_path[idx]
        image = Image.open(img_path).convert('RGB')
    
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        if self.dim==1:
            return {'input': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'input': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


class MujocoBuffer(Dataset):
    def __init__(
            self,
            data_folder: str,
            env: str,
            split: str,
            data_ratio,
            args = None,
            y_shift = None,
            div = None
    ):
        self.size = 0
        self.args=args
        self.state_dim = 0
        self.action_dim = 0

        self.states, self.actions = None, None
        self._load_dataset(data_folder, env, split, data_ratio)
        if self.args.y_norm == 'null':
            self.y_shift = None
            self.div = None
        elif self.args.y_norm in ['norm', 'std', 'scale']:
            if split == 'train':
                if self.args.y_norm == 'norm':
                    self.y_shift = np.mean(self.actions, axis=0)
                    centered_data = self.actions - self.y_shift
                    covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                    self.div = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
                elif self.args.y_norm == 'scale':
                    self.y_shift = np.min(self.actions, axis=0)
                    centered_data = self.actions - self.y_shift
                    self.div = np.diag( 1/(np.max(self.actions, axis=0)- self.y_shift))
                elif self.args.y_norm == 'std':
                    self.y_shift = np.mean(self.actions, axis=0)
                    centered_data = self.actions - self.y_shift
                    covariance_matrix = np.dot(centered_data.T, centered_data) / len(self.actions)
                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                    self.div = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ np.linalg.inv(eigenvectors)
            else:
                self.y_shift = y_shift
                centered_data = self.actions - self.y_shift
                self.div = div
            self.actions = centered_data @ self.div

            # self.actions = centered_data / self.div

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def _load_dataset(self, data_folder: str, env: str, split: str, data_ratio: float):
        file_name = '%s_%s.pkl' % (env, split)
        file_path = os.path.join(data_folder, file_name)
        try:
            with open(file_path, 'rb') as file:
                dataset = pickle.load(file)
                self.size = int(dataset['observations'].shape[0] * data_ratio)
                self.states = dataset['observations'][:self.size, :]
                self.actions = dataset['actions'][:self.size, :]
            print('Successfully load dataset from: ', file_path)
        except Exception as e:
            print(e)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        print(f"Dataset size: {self.size}; State Dim: {self.state_dim}; Action_Dim: {self.action_dim}.")

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim

    def get_theory_stats(self, center=False):
        mu = np.mean(self.actions, axis=0)
        if center:
            centered_actions = self.actions - mu
            Sigma = centered_actions.T @ centered_actions / centered_actions.shape[0]
        else:
            Sigma = self.actions.T @ self.actions / self.actions.shape[0]

        # Sigma_sqrt = sqrtm(Sigma)
        # eig_vals = np.linalg.eigvalsh(Sigma)
        eig_vals, eig_vecs = np.linalg.eigh(Sigma)
        sqrt_eig_vals = np.sqrt(eig_vals)
        Sigma_sqrt = eig_vecs.dot(np.diag(sqrt_eig_vals)).dot(np.linalg.inv(eig_vecs))

        min_eigval = eig_vals[0]
        max_eigval = eig_vals[-1]

        mu11 = Sigma[0, 0]
        mu12 = Sigma[0, 1]
        mu22 = Sigma[1, 1]

        sqrt = np.sqrt((mu22 - mu11) ** 2 + 4 * mu12 ** 2)
        gamma1 = (mu22 - mu11 + sqrt) / (2 * mu12)
        gamma2 = (mu22 - mu11 - sqrt) / (2 * mu12)

        return {
            'mu11': Sigma[0, 0],
            'mu12': Sigma[0, 1],
            'mu22': Sigma[1, 1],
            'min_eigval': min_eigval,
            'max_eigval': max_eigval,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'sigma11': Sigma_sqrt[0, 0],
            'sigma12': Sigma_sqrt[0, 1],
            'sigma21': Sigma_sqrt[1, 0],
            'sigma22': Sigma_sqrt[1, 1],
            'mu1': mu[0],
            'mu2': mu[1]
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return {
            'input': self._to_tensor(states),
            'target': self._to_tensor(actions)
        }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])