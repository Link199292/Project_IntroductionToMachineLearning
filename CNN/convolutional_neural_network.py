#This code has to be executed in Google Colab and the paths
#have to be changed according to your own paths

#Import

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
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
import cv2

import matplotlib.pyplot as plt
from google.colab import drive


#Mount your own GoogleDrive

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/')

#!unzip dataset.zip


#Define a class for images that allows to get the path and augment them

#Data Augmentation

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
        x = random.randint(0, self.img.shape[1] - int(width))
        y = random.randint(0, self.img.shape[0] - int(height))
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

os.chdir('/content/drive/MyDrive/IML/')



#Define the Convolutional Neural Network

class ConvNN(nn.Module):
  def __init__(self, in_channels=3, n_classes=10):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
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

        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),

        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.AdaptiveAvgPool2d((1, 1)),

        nn.Flatten(),
        nn.Linear(256, 128),
        nn.Linear(128, n_classes),
    )
    
  def forward(self, x):
    pred = self.cnn(x)
    return pred

mod = ConvNN()
if torch.cuda.is_available():
    mod.cuda()
summary(mod, (3, 64, 64))


#Data loader definition

class BasicDataset:
  def __init__(self, root, transform=None):
    self.transform = transform
    
    classes = os.listdir(root)
    
    self.data = []
    for y, class_ in enumerate(classes):
      folder = os.path.join(root, class_)
      for count, f in enumerate(os.listdir(folder)):
        f = os.path.join(folder, f)
        self.data.append((f, y))
        
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    f, y = self.data[index]
    x = PIL.Image.open(f)

    x = x.convert('RGB')

    if self.transform is not None:
      x = self.transform(x)
    return x, y

aug = T.Compose([
                       T.Resize((128, 128)),
                       T.ToTensor(),
                       T.Normalize([0.5], [0.5])
                       ])

train_path = '/content/drive/MyDrive/IML/dataset/training' #os.path.join(os.getcwd(),'' 'training')
test_path ='/content/drive/MyDrive/IML/dataset/validation/gallery' #os.path.join(os.getcwd(), 'validation', 'gallery')

train_dataset, test_dataset = BasicDataset(train_path, aug), BasicDataset(test_path, aug)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=64,
    num_workers=2,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=64,
    num_workers=2,
)



#training epochs function

#Training epoch

def train_one_epoch(device, model, loader, optimizer, scheduler):
  total = 0
  total_correct = 0
  
  model.train()
  for x, y in loader:

    x = x.to(device)
    y = y.to(device)

    out = model(x)
    loss = F.cross_entropy(out, y)

    pred = out.argmax(dim=1)

    total_correct += (pred == y).sum()
    total += y.size(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
  return total_correct / total



#eval epochs function

#Test epoch

def eval_one_epoch(device, model, loader):
  total = 0
  total_correct = 0

  model.eval()  # mode test

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      
      pred = out.argmax(dim=1)
      total_correct += (pred == y).sum()
      total += y.size(0)

  return total_correct / total



#define optimizer and scheduler

model = ConvNN(in_channels=3)


#optimizer = optim.RAdam(model.parameters(), lr=0.5, weight_decay=1e-5)

optimizer = torch.optim.SGD(
     model.parameters(),
     lr=0.03,
     momentum=0.9,
     weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40])


#Actual training

total_epochs = 100

device = torch.device('cuda:0')
model = model.to(device)

train_accs = []
test_accs = []
for epoch in trange(total_epochs):
  train_acc = train_one_epoch(device, model, train_loader, optimizer, scheduler)
  test_acc = eval_one_epoch(device, model, test_loader)
  train_accs.append(train_acc)
  test_accs.append(test_acc)
  
  print(f" Train acc {train_acc}, test acc {test_acc}")
  
  
#plotting results
 

x = [(1-l.item()) for l in train_accs]
x1 = [(1-l1.item()) for l1 in test_accs]

y= range(100)
  
plt.plot(y, x, label = "train error")
plt.plot(y, x1, label = "test error")
plt.legend()
plt.show()
