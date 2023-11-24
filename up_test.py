import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from PIL import Image
import numpy as np
import math
import time

from sklearn.model_selection import train_test_split

import upscale_util
import upscaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Our device is {}".format(device))

file_full = np.load("px256.npz")
full_size_img = file_full["face"]
print("Full faces loaded!")

file_small = np.load("px96_half.npz")
small_size_img = file_small["face"]
print("Half faces loaded!")

x_train, x_test, y_train, y_test = train_test_split(small_size_img, full_size_img,
                                                    test_size=0.2, random_state=42)

transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=[-10, 10], translate=[0.00, 0.08], scale=[0.65, 1.00], shear=5,
                            fill=(255, 255, 255)),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=[0.05, 0.08], ratio=[0.02, 0.05], p=0.5),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

training_set = upscale_util.UpDataSet(x_train, y_train, img_transform=transform_train)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True, num_workers=0)

test_set = upscale_util.UpDataSet(x_test, y_test, img_transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

up_model = upscaler.UNet()
up_model.to(device)

tensor_to_pillow = transforms.ToPILImage(mode="RGB")

up_model.load_state_dict(torch.load("best_upscaler_weights_upsampling.pth"))

with torch.no_grad():
    count = 1
    loss = 0
    for x, y in test_loader:
        if count % 1000 == 0:
            x = x.to(device)
            y = y.to(device)
            outputs = up_model(x)
            out_x = x.cpu()[0]
            out_y = y.cpu()[0]
            out = outputs.cpu()[0]
            out_x = tensor_to_pillow(out_x)
            out_y = tensor_to_pillow(out_y)
            out = tensor_to_pillow(out)
            out_x.show()
            out_y.show()
            out.show()
        count += 1


"""
my_encoder = upscaler.Encoder()
my_encoder.to(device)
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    inferences = my_encoder(x)
    print(inferences[-1].size())
    break
"""
