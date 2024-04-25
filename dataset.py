import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageTargetDataset(Dataset):

    def __init__(self, images_dir, targets_dir, transform=None):
        """
        Args:
            images_dir (str): Path to the directory containing image files.
            targets_dir (str): Path to the directory containing target files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.targets_dir = targets_dir
        self.transform = transform

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

        return {'image': image, 'target': torch.tensor(target[[0, 10]], dtype=torch.float)}
    

class H5Dataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.h5')]

        self.data = []
        self.labels = []

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                images_dataset = file['rgb']   
                labels_dataset = file['targets']   
                for i in range(len(images_dataset)):
                    self.data.append(images_dataset[i])
                    self.labels.append(labels_dataset[i])
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image), torch.tensor(label)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])