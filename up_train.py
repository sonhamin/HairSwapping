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
    transforms.RandomHorizontalFlip(),
    upscale_util.MyRotateTransform(angles=(0, 90, 180, 270)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

training_set = upscale_util.UpDataSet(x_train, y_train, img_transform=transform_train)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True, num_workers=0)

test_set = upscale_util.UpDataSet(x_test, y_test, img_transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)

up_model = upscaler.UNet()
up_model.to(device)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(up_model.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer = torch.optim.SGD(up_model.parameters(), lr=0.0005, momentum=0.9)


start_epoch = 1
best_test_loss = math.inf

"""
try:
    up_model.load_state_dict(torch.load("best_upscaler_weights.pth"))
    best_test_loss = 0.046
    start_epoch = 3
except FileNotFoundError as e:
    print(str(e))
"""

tensor_to_pillow = transforms.ToPILImage(mode="RGB")


num_epoch = 10
print("{} batches feeding in!".format(len(train_loader)))

for epoch in range(start_epoch, num_epoch + 1):
    last_start = time.time()
    print("Epoch " + str(epoch) + " in progress...")
    up_model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        inferences = up_model(x)

        optimizer.zero_grad()
        cur_loss = criterion(inferences, y)
        cur_loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += cur_loss.item()

    running_loss /= len(train_loader)
    end_time = time.time()
    print("Epoch {0}: training_loss: {1}, epoch_training_time: {2}s. ".format(epoch, running_loss,
                                                                              end_time - last_start))

    up_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            inferences = up_model(x)
            cur_loss = criterion(inferences, y)
            test_loss += cur_loss
        test_loss /= len(test_loader)
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(up_model.state_dict(), f="best_upscaler_weights.pth")

    try:
        if epoch % 5 == 0 or epoch == 1:
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
    except Exception as e:
        print(str(e))
    print("Epoch {0}: test_loss: {1}, best_loss: {2}".format(epoch, test_loss, best_test_loss))

torch.save(up_model.state_dict(), f="final_upscaler_weights.pth")

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
