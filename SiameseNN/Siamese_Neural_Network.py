#This code is meant to be run into Google Colab and the paths
#should be changed accordingly to your data paths

import os
from tqdm import tqdm, trange
import PIL.Image

import random
import numpy as np
import imutils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchsummary import summary
import cv2

import matplotlib.pyplot as plt
from google.colab import drive


drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/machine_learning_project')
#!unzip 'dataset.zip'





#Class definition for reading images and augmentation

class MImage:
    """takes the position of an image, opens it and allows to augment it multiple times
       by storing it in the same folder
    """
    def __init__(self, location, times=5):

        self.file_name = os.path.basename(location)
        self.location = os.path.abspath("/".join(os.path.abspath(location).split('/')[:-1]))
        self.times = times + 1
        self.img = cv2.imread(os.path.join(self.location, self.file_name))


    def crop(self, scale=0.5):
        height, width = int(self.img.shape[0]*scale), int(self.img.shape[1]*scale)
        center =  self.img.shape / 2
        x = center[1] - width / 2
        y = center[0] - height / 2
        cropped = self.img[y:y+height, x:x+width]
        resized = cv2.resize(cropped, (self.img.shape[1], self.img.shape[0]))
        cv2.imwrite(os.path.join(self.location, f'cropped_' + self.file_name), resized)


    def color_jitter(self):

        # Brightness
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            os.chdir(self.location)
            cv2.imwrite(os.path.join(self.location, f'bjitter_' + self.file_name), image)

        # Saturation
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(self.location, f'sjitter_' + self.file_name), image)

        # Contrast
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(self.img)
        dummy = dummy * (contrast/127+1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        image = np.uint8(dummy)
        cv2.imwrite(os.path.join(self.location, f'cjitter_' + self.file_name), image)


    def add_noise(self):

        #Gaussian
        image = self.img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean,st,image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image,gauss)
        cv2.imwrite(os.path.join(self.location, f'ngaussian_{i}' + self.file_name), image)

        #Salt & Pepper
        image = self.img.copy() 
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255            
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        cv2.imwrite(os.path.join(self.location, f'nsap_{i}' + self.file_name), image)


    def filtering(self):
        fsize = 9

        # Blur
        image = self.img.copy()
        cv2.imwrite(os.path.join(self.location, f'fblur_' + self.file_name), cv2.blur(image, (fsize,fsize)))

        #Gaussian
        image = self.img.copy()
        cv2.imwrite(os.path.join(self.location, f'fgauss_' + self.file_name), cv2.GaussianBlur(image, (fsize, fsize), cv2.BORDER_DEFAULT))

        #Median
        image = self.img.copy()
        cv2.imwrite(os.path.join(self.location, f'fmedian_' + self.file_name), cv2.medianBlur(image, fsize))

    def rotate(self):
        idx = 1
        list_degree = []
        while idx < self.times:
            image = self.img.copy()
            deg = random.randint(-179, 180)
            if deg == 0 or deg in list_degree:
                continue
            rotated = imutils.rotate_bound(image, deg)
            if deg > 0:
                cv2.imwrite(os.path.join(self.location, f'rrotated_{deg}' + self.file_name), rotated)
            else:
                cv2.imwrite(os.path.join(self.location, f'lrotated_{deg}' + self.file_name), rotated)
            list_degree.append(deg)
            idx += 1

    def augment(self):
        self.rotate()
        self.filtering()
        self.add_noise()
        self.color_jitter()
        self.crop()
        
       
      
      
      
#Data augmentation process

os.chdir('/content/drive/MyDrive/machine_learning_project')

path = os.path.join(os.getcwd(), 'training')

training = {}
 
for folders, _, files in os.walk(path): 
  for f in files: 
    img = os.path.join(folders, f)
    curr_class = os.path.basename(folders).split('(')[-1].rstrip(')')
    if curr_class not in training:
      training[curr_class] = [img]
    else:
      training[curr_class].append(img)

for i in training:
  for j in training[i]:
    curr_img = MImage(j)
    print(curr_img.location)
    curr_img.augment()

os.chdir('/content/drive/MyDrive/machine_learning_project')





#Define Siamese Network

class SiameseNN(nn.Module):
  def __init__(self, in_channels=1, n_classes=10):
    super().__init__()
    self.siamese = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),

                nn.AdaptiveAvgPool2d((1, 1)),

                nn.Flatten(),
                nn.Linear(256, n_classes),
                )
    
  def forward_once(self, x):
    output = self.siamese(x)
    return output

  def forward(self, input1, input2):
    output1 = self.forward_once(input1)
    output2 = self.forward_once(input2)

    return output1, output2
  
  
  
  

#Define Loss function (Contrastive Loss Function)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive
    
 



os.chdir('/content/drive/MyDrive/machine_learning_project')

train_transform = T.Compose([
                       T.Resize((256, 256)),
                       T.ToTensor(),
                       T.Normalize([0.5], [0.5])
])

test_transform = T.Compose([
                       T.Resize((256, 256)),
                       T.ToTensor(),
                       T.Normalize([0.5], [0.5])
])

train_path = os.path.join(os.getcwd(), 'training')
test_path = os.path.join(os.getcwd(), 'validation', 'gallery')

train_dataset = BasicDataset(train_path, transform=train_transform)
test_dataset = BasicDataset(test_path, transform=test_transform)





#Define Data Loader

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=32,
    num_workers=2,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=32,
    num_workers=2
)





#Define function to train one epoch

def train_one_epoch(device, model, loader, optimizer, scheduler, criterion):
  total = 0
  total_correct = 0

  model.train()  # set model to train mode

  for i, (img0, img1, label) in enumerate(loader, 0):
    # move data to the correct device e.g. GPU
    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

    # get output of the model
    output1, output2 = model(img0, img1)
    # compute the loss
    loss_contrastive = criterion(output1, output2, label)
    pred = [int(i > 0.5) for i in F.pairwise_distance(output1, output2)]
    total_correct += sum([int(i == j) for i, j in zip(pred, label)])
    total += len(pred)
    #zeros the grads
    optimizer.zero_grad()
    # backprops the loss
    loss_contrastive.backward()
    # one step with the optimizer
    optimizer.step()
    # one step with scheduler
    scheduler.step()

  return total_correct / total





#Define function to test one epoch

def eval_one_epoch(device, model, loader, criterion):
  total = 0
  total_correct = 0
  model.eval()  # set model to eval mode
  # we don't need gradients when doing evaluation
  with torch.no_grad():
     for i, (img0, img1, label) in enumerate(loader, 0):
      # move data to the correct device
      img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

      # output of the model
      output1, output2 = model(img0, img1)

      # predictions
      loss_contrastive = criterion(output1, output2, label)
      pred = [int(i > 0.5) for i in F.pairwise_distance(output1, output2)]
      total_correct += sum([int(i == j) for i, j in zip(pred, label)])
      total += len(pred)

  return total_correct / total





#Train the model

model = SiameseNN(in_channels = 3)
criterion = ContrastiveLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.03,
    momentum=0.9,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60, 80])

total_epochs = 100

device = torch.device('cuda:0')
model = model.to(device)

train_losses = []
test_losses = []

for epoch in trange(total_epochs):
  train_loss = train_one_epoch(device, model, train_loader, optimizer, scheduler, criterion)
  test_loss = eval_one_epoch(device, model, test_loader, criterion)
  train_losses.append(train_loss)
  test_losses.append(test_loss)

  print(f"Train err: {1 - train_loss}, Test err: {1 - test_loss}")
