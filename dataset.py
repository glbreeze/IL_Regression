import os
import h5py
import pickle 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
            return {'image': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'image': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


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
            return {'image': image, 'target': torch.tensor(target[0], dtype=torch.float)}
        elif self.dim==2:
            return {'image': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}


class NumpyDataset(Dataset):
    def __init__(self, images_file, targets_file, transform=None):
        self.images = np.load(images_file)
        self.targets = np.load(targets_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        targets = self.targets[idx, [0, 10]]  #steer and speed
        if self.transform:
            image = self.transform(image)

        return {'image': torch.tensor(image, dtype=torch.float).permute(0, 2, 1),  # Adjust for PyTorch: [C, H, W]
                'target': torch.tensor(targets, dtype=torch.float)}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])