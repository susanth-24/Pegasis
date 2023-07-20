import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
import os

class NYUV2Dataset(torch.utils.data.Dataset):
    '''
    This is the dataset loader for that will be used in training the depth model
    Do change the image size of the depth image if needed.
    '''
    def __init__(self, csv_file, base_dir, output_shape, transform=None):
        self.data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.output_shape = output_shape
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.base_dir, self.data.iloc[idx, 0])
        depth_path = os.path.join(self.base_dir, self.data.iloc[idx, 1])
        
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')
        
        depth_image = depth_image.resize((self.output_shape[3], self.output_shape[2]))
        
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
        
        depth_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        depth_image = depth_transform(depth_image)

        return rgb_image, depth_image