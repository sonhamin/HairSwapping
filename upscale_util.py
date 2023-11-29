import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import vgg19

class UpDataSet(Dataset):
    def __init__(self, small_img, full_img, img_transform=None):
        self.small_img = small_img
        self.full_img = full_img
        self.img_transform = img_transform

    def __len__(self):
        return self.full_img.shape[0]

    def __getitem__(self, index):
        small = Image.fromarray(self.small_img[index].astype(np.uint8), mode="RGB")
        full = Image.fromarray(self.full_img[index].astype(np.uint8), mode="RGB")
        if self.img_transform:
            random.seed(42)
            small = self.img_transform(small)
            full = self.img_transform(full)
        return small, full


class MyRotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained=True).features)[:36]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {2, 7, 12, 21, 30}:
                results.append(x)
        return results
