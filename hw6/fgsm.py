import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

seed = 726
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda")
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []
        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return 200
class Attacker:
    def __init__(self, img_dir, output_dir, label):
        self.model = models.densenet121(pretrained = True) 
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std, inplace=False)
                    ])
        self.dataset = Adverdataset(img_dir, label, transform)
        self.loader = DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)
        self.img_dir = img_dir
        self.output_dir = output_dir
        
    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilon):
        adv_examples = []
        fail, success = 0, 0
        for idx, (data, target) in enumerate(self.loader):
            orig_img = data
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                perturbed_data = data
            else:
                loss = F.cross_entropy(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target.item():
                fail += 1
            else:
                success += 1

            img = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
            img = img.squeeze().detach().cpu().numpy()
            img[img < 0] = 0
            img[img > 1] = 1
            img = (img*255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(self.output_dir, '{:03d}.png'.format(idx)))

        final_acc = (fail / (success + fail))
        
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, success+fail, final_acc))
        return adv_examples, final_acc
data_dir = sys.argv[1] 
output_dir = sys.argv[2]
df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
df = df.loc[:, 'TrueLabel'].to_numpy()
label_name = pd.read_csv(os.path.join(data_dir, "categories.csv"))
label_name = label_name.loc[:, 'CategoryName'].to_numpy()
attacker = Attacker(os.path.join(data_dir, 'images'), output_dir, df)

epsilons = [0.1]

accuracies, examples = [], []
for eps in epsilons:
    ex, acc = attacker.attack(eps)
    accuracies.append(acc)
    examples.append(ex)
