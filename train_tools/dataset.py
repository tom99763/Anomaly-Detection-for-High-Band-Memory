import torch
import lightning as L
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split as ttp
import os

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class HBMDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
    def __getitem__(self, idx):
        _dir = self.data_dir[idx]
        img = rgb_loader(_dir)
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.data_dir)

class HBMDataModule(L.LightningDataModule):
    def __init__(self, opt, transform):
        super().__init__()
        self.batch_size = opt.batch_size
        self.transform = transform
        train_dir = f'{opt.dataset_dir}/train/defect'
        test_dir = f'{opt.dataset_dir}/test/defect'
        train_dir = list(map(lambda x: f'{train_dir}/{x}',os.listdir(train_dir)))
        self.train_dir, self.val_dir = ttp(train_dir, test_size=opt.val_ratio, random_state=0)
        self.test_dir = list(map(lambda x: f'{test_dir}/{x}',os.listdir(test_dir)))

        self.ds_train = HBMDataset(self.train_dir, self.transform)
        self.ds_val = HBMDataset(self.val_dir, self.transform)
        self.ds_test = HBMDataset(self.test_dir, self.transform)
    '''
    def setup(self, stage: str):
        self.ds_train = HBMDataset(self.train_dir, self.transform)
        self.ds_val = HBMDataset(self.val_dir, self.transform)
        self.ds_test = HBMDataset(self.test_dir, self.transform)
    '''

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)