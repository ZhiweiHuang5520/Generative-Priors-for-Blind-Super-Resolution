import numpy as np
import random
from torchvision.datasets.folder import default_loader
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, HRroot, LRroot, sf = 2, crop_size = 64, split = 'Train', val_crop = 256):
        super(MyDataset, self).__init__()

        self.HRroot = Path(HRroot)
        self.LRroot = Path(LRroot)
        if split == 'Train':
            self.transforms = transforms.Compose([
                                                  transforms.CenterCrop([crop_size , crop_size ]),
                                                # transforms.RandomHorizontalFlip(),
                                                #   transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.target_transforms = transforms.Compose([
                                                  transforms.CenterCrop([crop_size * sf, crop_size * sf]),
                                                # transforms.RandomHorizontalFlip(),
                                                #   transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop([val_crop, val_crop]),
                # transforms.RandomHorizontalFlip(),
                #   transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.target_transforms = transforms.Compose([
                transforms.CenterCrop([val_crop * sf, val_crop * sf]),
                # transforms.RandomHorizontalFlip(),
                #   transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        

        self.HRimgs = sorted([f for f in self.HRroot.iterdir() if f.suffix in ['.jpeg','.jpg','.png']])
        self.LRimgs = sorted([f for f in self.LRroot.iterdir() if f.suffix in ['.jpeg', '.jpg', '.png']])

    def __len__(self):
        return len(self.HRimgs)

    def __getitem__(self, idx):
        HRimg = default_loader(self.HRimgs[idx])
        LRimg = default_loader(self.LRimgs[idx])

        
        if self.transforms is not None:
            LRimg = self.transforms(LRimg)

        if self.target_transforms is not None:
            HRimg = self.target_transforms(HRimg)

        return LRimg, HRimg


