import sys
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

random.seed(726)
np.random.seed(726)
torch.manual_seed(726)
torch.cuda.manual_seed(726)
torch.backends.cudnn.deterministic = True


gpu = torch.device("cuda:0")
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

workspace_dir = sys.argv[1]
output_dir = sys.argv[2]
ckptpath = './p3.pt'

train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
test_x = readfile(os.path.join(workspace_dir, "testing"), False)

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(45), #隨機旋轉圖片
    transforms.RandomAffine( degrees = 0, translate = (0.05, 0.05), scale = (0.9,1.1), shear = 5, resample = False, fillcolor = 0),
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
])

#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 1, 1),      # [32, 127, 127]

            nn.Conv2d(32, 64, 3, 1, 1),  # [64, 127, 127]
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2,2,0),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model = Classifier().cuda()
checkpoint = torch.load(ckptpath)
model.load_state_dict(checkpoint)
model.eval()
val_label = np.zeros((len(val_y)), dtype = np.uint8)

with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        val_label[i * batch_size : (i + 1) * batch_size] = np.argmax(val_pred.cpu().data.numpy(), axis = 1)
    
classes = ('Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit')
cm = confusion_matrix(val_y, val_label)
cm = cm.astype(float) / np.sum(cm, axis = 1)[:, np.newaxis]   ## (11,) --> (11, 1)
    
plt.figure(figsize = (10, 10))
plt.title('Confusion Matrix')
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation = 45)
plt.yticks(tick_marks, classes)

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = 'white' if i == j else ('red' if cm[i, j] > 0.1 else 'black')
    plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment = 'center', color = color)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
    
plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.savefig(os.path.join(output_dir, "Confusion_matrix.png")) 
plt.close()
