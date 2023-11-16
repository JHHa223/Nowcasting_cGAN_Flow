import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, is_train=0, transforms_=None, mode="train"):
        self.is_train = is_train
        self.transform = transforms.Compose(transforms_)

        if is_train == 0:
            self.path = glob.glob('/data0/jhha223/pix2pix/train_set/*/*.npy') #training dataset
        else:
            self.path = glob.glob('/data0/jhha223/pix2pix/val_set/*/*.npy') #validation dataset

    def __getitem__(self, index):
        img = np.load(self.path[index])
        img_A = Image.fromarray(img[:,:,:3],"RGB") #image produced by optical flow
        img_B = Image.fromarray(img[:,:,3:],"RGB") #ground truth
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.path)
