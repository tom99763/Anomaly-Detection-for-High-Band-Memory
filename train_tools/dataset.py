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
    def __init__(self, data_dir, label,  transform):
        super().__init__()
        self.data_dir = data_dir
        self.label = label
        self.transform = transform
    def __getitem__(self, idx):
        _dir = self.data_dir[idx]
        label = self.label[idx]
        img = rgb_loader(_dir)
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.data_dir)

class HBMDataModule(L.LightningDataModule):
    def __init__(self, opt, transform):
        super().__init__()
        self.batch_size = opt.batch_size
        self.transform = transform
        train_dir = f'{opt.dataset_dir}/train'
        test_dir = f'{opt.dataset_dir}/test'
        #train dir
        train_pass_dir = list(map(lambda x: f'{train_dir}/Pass/{x}',
                                  os.listdir(f'{train_dir}/Pass')))

        '''
        try:
            train_reject_dir = list(map(lambda x: f'{train_dir}/Reject/{x}',
                                        os.listdir(f'{train_dir}/Reject')))
        except:
            train_reject_dir = []
        '''

        train_reject_dir = []
        self.train_dir = train_pass_dir + train_reject_dir
        self.train_label = [0] * len(train_pass_dir) + [1] * len(train_reject_dir)
        #self.train_dir, self.val_dir, self.train_label, self.val_label = ttp(train_dir, train_label,
        #                                   test_size=opt.val_ratio, random_state=0)

        #test dir
        test_pass_dir = list(map(lambda x: f'{test_dir}/Pass/{x}',
                                 os.listdir(f'{test_dir}/Pass')))
        test_reject_dir = list(map(lambda x: f'{test_dir}/Reject/{x}',
                                   os.listdir(f'{test_dir}/Reject')))
        self.test_dir = test_pass_dir + test_reject_dir
        self.test_label = [0] * len(test_pass_dir) + [1] * len(test_reject_dir)

        #dataset
        self.ds_train = HBMDataset(self.train_dir, self.train_label, self.transform)
        #self.ds_val = HBMDataset(self.val_dir, self.val_label, self.transform)
        self.ds_test = HBMDataset(self.test_dir, self.test_label, self.transform)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          num_workers=19, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=32,
                          num_workers=19, persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=32,
                          num_workers=19, persistent_workers=True, shuffle=True)