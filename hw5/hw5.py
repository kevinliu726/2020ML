import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import random
import cv2
import shap
random.seed(726)
np.random.seed(726)
torch.manual_seed(726)
torch.cuda.manual_seed(726)
torch.backends.cudnn.deterministic = True
 
ckptpath = './best.pt'
workspace_dir = sys.argv[1]
output_dir = sys.argv[2]

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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

class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        
        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128,128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomAffine( degrees = 0, translate = (0.05, 0.05), scale = (0.9,1.1), shear = 5, resample = False, fillcolor = 0),
            transforms.ToTensor(), 
        ])

        evalTransform = transforms.Compose([
            transforms.Resize(size=(128,128)),                                    
            transforms.ToTensor(),
        ])

        self.transform = trainTransform if mode == 'train' else evalTransform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

model = Classifier()
checkpoint = torch.load(ckptpath)
model.load_state_dict(checkpoint)
train_paths, train_labels = get_paths_labels(os.path.join(workspace_dir, 'training'))

train_set = FoodDataset(train_paths, train_labels, mode='eval')
train_loader = DataLoader(train_set, batch_size = 128, shuffle = False)

def denormalize(img):
    convert_img = img*0.5 + 0.5
    return convert_img 

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    model.cuda()
    x = x.cuda()

    x.requires_grad_()
  
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

#Saliency map
img_indices = [425, 6969, 1010, 7777]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img.permute(1, 2, 0).numpy())
plt.savefig(os.path.join(output_dir, './task1.jpg'))
plt.close()

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    model.eval()
    model.cuda()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output
  
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    model(x.cuda())
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
    x = x.cuda()
    x.requires_grad_()
    optimizer = Adam([x], lr=lr)
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
    
        objective = -layer_activations[:, filterid, :, :].sum()
    
        objective.backward()
        optimizer.step()
    filter_visualization = x.detach().cpu().squeeze()[0]
    return filter_activations, filter_visualization

img_indices = [1111, 8585, 6969, 7777]
images, labels = train_set.getbatch(img_indices)

fa, fv = filter_explaination(images, model, cnnid=10, filterid=0, iteration=100, lr=0.1)
fa2, fv2 = filter_explaination(images, model, cnnid=10, filterid=2, iteration=100, lr=0.1)
fa3, fv3 = filter_explaination(images, model, cnnid=10, filterid=4, iteration=100, lr=0.1)
fa4, fv4 = filter_explaination(images, model, cnnid=10, filterid=6, iteration=100, lr=0.1)

fig, axs = plt.subplots(1, 4, figsize=(15,8))
axs[0].imshow(normalize(fv.permute(1,2,0)))
axs[1].imshow(normalize(fv2.permute(1,2,0)))
axs[2].imshow(normalize(fv3.permute(1,2,0)))
axs[3].imshow(normalize(fv4.permute(1,2,0)))
plt.savefig(os.path.join(output_dir, './task2_v.jpg'))
plt.close()

# 畫出 filter activations
fig, axs = plt.subplots(5, len(img_indices)+1, figsize=(16,20))
axs[0][0].axis('off')
for i, img in enumerate(images):
  axs[0][i+1].imshow(img.permute(1, 2, 0))
axs[1][0].imshow(normalize(fv.permute(1,2,0)))
axs[2][0].imshow(normalize(fv2.permute(1,2,0)))
axs[3][0].imshow(normalize(fv3.permute(1,2,0)))
axs[4][0].imshow(normalize(fv4.permute(1,2,0)))
for i, img in enumerate(fa):
  axs[1][i+1].imshow(normalize(img))
for i, img in enumerate(fa2):
  axs[2][i+1].imshow(normalize(img))
for i, img in enumerate(fa3):
  axs[3][i+1].imshow(normalize(img))
for i, img in enumerate(fa4):
  axs[4][i+1].imshow(normalize(img))
plt.savefig(os.path.join(output_dir, './task2_a.jpg'))

plt.close()

model = Classifier().cuda()
checkpoint = torch.load(ckptpath)
model.load_state_dict(checkpoint)

batch = next(iter(train_loader))
images, _ = batch
n_background_images = 50
background = images[:n_background_images] .cuda()
e = shap.DeepExplainer(model, background)
n_test_images = 5
test_images = images[55 + n_background_images : 55 + n_background_images+n_test_images]
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
for idx in range(n_test_images):
    test_numpy[idx] = denormalize(test_numpy[idx])
shap.image_plot(shap_numpy, test_numpy,show=False)
plt.savefig(os.path.join(output_dir, './shap.jpg'))

def predict(input):
    model.eval()
    input = torch.FloatTensor(input).permute(0,3,1,2)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    return slic(input, n_segments = 100, compactness = 1, sigma = 1)

img_indices = [1,425,1504,1111,8585,8586,4002,4010]
images, labels = train_set.getbatch(img_indices)
fix, axs = plt.subplots(2,8, figsize=(30,16))

for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    explainer = lime_image.LimeImageExplainer()
    np.random.seed(0)
    explaination = explainer.explain_instance(image=x, top_labels=11, classifier_fn=predict, segmentation_fn=segmentation)
    lime_img, mask = explaination.get_image_and_mask(
            label=label.item(),
            positive_only=False,
            hide_rest=False,
            num_features=11,
            min_weight=0.05
            )
    axs[1][idx].imshow(lime_img)
for idx, img in enumerate(images):
    axs[0][idx].imshow(img.permute(1, 2, 0).numpy())

plt.savefig(os.path.join(output_dir, './task3.jpg'))
plt.close()
