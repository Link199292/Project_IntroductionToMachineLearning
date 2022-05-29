#This code is inteded to be run into Google Colab and the paths
#to be changed accordingly to your data path


#!pip install tensorflow
#!pip install keras
#!pip install tensorflow-gpu

# standard imports
from google.colab import drive
import numpy as np
import os
import random
import time

# utils
from tqdm import tqdm
from numpy.linalg import norm
from matplotlib import rcParams
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# tensorflow
import tensorflow as tf
from keras import layers
from keras.preprocessing import image
from keras.applications.resnet import ResNet152, preprocess_input, ResNet50
from keras import regularizers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from keras.models import load_model
import cv2

import PIL
import matplotlib.pyplot as plt
from math import ceil
import pickle


drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/machine_learning_project')





#Define Class for image reading and augmenting

## Data Augmentation


class MImage:
  """takes the position of an image, opens it and allows to augment it multiple times
  by storing it in the same folder
  """
  def __init__(self, location):
    self.file_name = os.path.basename(location)
    self.location = os.path.abspath("/".join(os.path.abspath(location).split('/')[:-1]))
    self.img = cv2.imread(os.path.join(self.location, self.file_name))
  
  def add_noise(self):
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
    cv2.imwrite(os.path.join(self.location, f'nsap_' + self.file_name), image)

  def vertical_flip(self):
    image = self.img.copy()
    v_flip= cv2.flip(image, 0)
    cv2.imwrite(os.path.join(self.location, f'v_flip_' + self.file_name), v_flip)

  def horizontal_flip(self):
    image = self.img.copy()
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(self.location, f'h_flip' + self.file_name), h_flip)
  
  def both_flip(self):
    image = self.img.copy()
    b_flip = cv2.flip(image, -1)
    cv2.imwrite(os.path.join(self.location, f'b_flip' + self.file_name), b_flip)

  def rotate_clock(self):
    image = self.img.copy()
    b_clock = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(self.location, f'90c' + self.file_name), b_clock)

  def rotate_counterclock(self):
    image = self.img.copy()
    b_counter = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(self.location, f'90cc' + self.file_name), b_counter)

  def augment(self, q=None):
    #if q is set to some number, that number is used as an indicator of the number
    #of transformations to randomly apply to the current image
    if not q:
      self.add_noise()
      self.vertical_flip()
      self.horizontal_flip()
      self.both_flip()
      self.rotate_clock()
      self.rotate_counterclock()
    else:
      print('NOT IMPLEMENTED')


    
    
    
#Perform rebalancing of the class according to the dictionary 'quantities'

quantities = {'training' : 500,
              'gallery' : 125,
              'query' : 25}

def count_images(path):
  counts = {}
  for i, j, k in os.walk(path):
    if i != path:
      for n in k:
        curr_folder = os.path.join(i)
        if curr_folder not in counts:
          counts[curr_folder] = 1
        else:
          counts[curr_folder] += 1
  return counts

#keep the number of transformation constant (to 6), just change the number of files to pick (each cycle add 6)

def augment(n_images, thresh, fold_name):
  lst_of_imgs = [os.path.join(fold_name, n) for i, j, k in os.walk(fold_name) for n in k] #take the full list of images in the current class folder
  n_augmented = 0
  idx = 0
  while n_augmented + n_images <= thresh:
    try:
      curr_img = MImage(lst_of_imgs[idx])
      curr_img.augment()
      idx += 1
      n_augmented += 6
    except:
      break
  
def reduce(n_images, thresh, fold_name):
  to_del_lst = [] #to delete images
  lst_of_imgs = [os.path.join(fold_name, n) for i, j, k in os.walk(fold_name) for n in k] #list of images in folder
  to_del_counter = n_images - thresh #30 - 25 = 5
  counter = 0
  while counter < to_del_counter:
    to_del_lst.append(lst_of_imgs[counter])
    counter += 1
  for img in to_del_lst:
    os.remove(img)
    print(f'{img} has been removed')

def augment_reduce(diz, mode='training'):

  thresh = quantities[mode]
  for i in diz:
    if diz[i] < thresh:
      augment(diz[i], thresh, i)
    else:
      reduce(diz[i], thresh, i)


diz = count_images(training_path)
augment_reduce(diz)



#Check the result of the rebalancing:
#Up to approximation due to the algo, we should get ~ 500 images in training (augmented),
# ~ 125 images in gallery (removing some of them) and ~ 25 in query (removing some of them)


training_path = '/content/drive/MyDrive/machine_learning_project/images/training'

def count_images(path):
  counts = {}
  for i, j, k in os.walk(path):
    if i != path:
      for n in k:
        curr_folder = os.path.join(i)
        if curr_folder not in counts:
          counts[curr_folder] = 1
        else:
          counts[curr_folder] += 1
  return counts

count_images(query_path)





#ResNet Initialization

resnet = ResNet152( #put ResNet50 to use that version
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3))



#count the number of classes

for i in os.walk('/content/drive/MyDrive/machine_learning_project/images/training'):
    n_classes = len(i[1])
    break
    
    
    
#Unfreeze the 5th layer
    
resnet.trainable = False

for layer in resnet.layers:
  if 'conv5' in layer.name:
    layer.trainable=True

inputs = tf.keras.Input((256, 256, 3))
x = resnet(inputs, training=False)
x = layers.Flatten(name='flatten_39')(x)
x = layers.Dense(units=n_classes, activation='softmax', kernel_regularizer=regularizers.L1L2())(x)

model = tf.keras.Model(inputs, x)
 
  
  
  
  
#Define the Data Loader
  
os.chdir('/content/drive/MyDrive/machine_learning_project/images')
data_path = os.getcwd()

train_path = os.path.join(data_path, 'training')
test_path = os.path.join(data_path, 'validation', 'gallery')

batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    subset='training',
    validation_split=0.2)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    labels='inferred',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=False,
    seed=42,
    subset='validation',
    validation_split=0.2)





#Compile and fit the model

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=test_dataset, epochs=2)


#Truncate the last layer

model2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name='flatten_39').output)

#To Save the model

model2.save('/content/drive/MyDrive/machine_learning_project/images/models/ResNet50_features')
model2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(name='flatten_39').output)

#To Load the model

model2 = load_model('/content/drive/MyDrive/machine_learning_project/images/models/ResNet50_features')


#Pick all query images and gallery images, transform them based on the feature extractor
#Code enhanced by Victor Turrisi Da Costa in order to transform data in batches

os.chdir('/content/drive/MyDrive/machine_learning_project/images')
data_path = os.getcwd()


query_path = os.path.join(data_path, 'validation', 'query')
gallery_path = os.path.join(data_path, 'validation', 'gallery')


def pick_all_queries(query_path, mod):
  query = []
  batch = []
  for i, j, k in os.walk(query_path):
    for x in tqdm(k):
      new_path = os.path.join(query_path, i, x)
      img = image.load_img(new_path, target_size=(256,256), color_mode='rgb')
      img = np.array(img)
      img = np.expand_dims(img, axis = 0)
      img = preprocess_input(img)
      class_name = new_path.split('/')[-2]
      batch.append((img, new_path, class_name))
      if len(batch) % 64 == 0:
        data = np.concatenate([b[0] for b in batch])
        paths = [b[1] for b in batch]
        names = [b[2] for b in batch]
        features = mod.predict(data)
        for f, p, n in zip(features, paths, names):
          query.append((f, p, n))
        batch = []
  if batch:
    data = np.concatenate([b[0] for b in batch])
    paths = [b[1] for b in batch]
    names = [b[2] for b in batch]
    features = mod.predict(data)
    for f, p, n in zip(features, paths, names):
      query.append((f, p, n))
  return query


def pick_all_galleries(gallery_path, mod):
  galleries = []
  batch = []
  for i, j, k in os.walk(gallery_path):
    for x in tqdm(k):
      new_path = os.path.join(gallery_path, i, x)
      img = image.load_img(new_path, target_size=(256,256), color_mode='rgb')
      img = np.array(img)
      img = np.expand_dims(img, axis = 0)
      img = preprocess_input(img)
      class_name = new_path.split('/')[-2]
      batch.append((img, new_path, class_name))
      if len(batch) % 64 == 0:
        data = np.concatenate([b[0] for b in batch])
        paths = [b[1] for b in batch]
        names = [b[2] for b in batch]
        features = mod.predict(data)
        for f, p, n in zip(features, paths, names):
          galleries.append((f, p, n))
        batch = []
  if batch:
    data = np.concatenate([b[0] for b in batch])
    paths = [b[1] for b in batch]
    names = [b[2] for b in batch]
    features = mod.predict(data)
    for f, p, n in zip(features, paths, names):
      galleries.append((f, p, n))
  return galleries

all_queries = pick_all_queries(query_path, model2)
all_galleries = pick_all_galleries(gallery_path, model2)





#To compute cosine similarity between a query images and all the provided gallery
#images and rank them from closest to further.
#After that, a metric for the performance is computed, according to the competition one.

def compute_cos_similarity(query_img, gallery_images):
  gallery = [(i[0], i[1], i[2]) for i in gallery_images]
  results = []
  for i in gallery:
    compute = cosine_similarity(query_img[0].reshape(1, -1), i[0].reshape(1, -1))
    results.append((compute, i[1], query_img[-1], i[-1]))
  results.sort(reverse=True, key = lambda x : x[0])
  return results


def submit_results(all_queries, all_galleries, n_queries=30):
  diz = {}
  q = 0
  while q < n_queries:
    curr_query = all_queries[np.random.randint(0, len(all_queries) - 1)]
    res = compute_cos_similarity(curr_query, all_galleries)
    top_10 = [i[1] for i in res[:10]]
    if curr_query[1] not in diz:
      diz[curr_query[1]] = top_10
      q += 1
    else:
      continue
  return diz

submitted_results = submit_results(all_queries, all_galleries, n_queries=5)


def metric_evaluation(all_queries, all_galleries, n_queries=30):

  metrics = {'top_1' : 0,
             'top_3' : 0,
             'top_10' : 0}

  for i in range(n_queries):
    curr_query = all_queries[np.random.randint(0, len(all_queries) - 1)]
    results = compute_cos_similarity(curr_query, all_galleries)

    top_10 = [i for i in results[:10]]

    #top1
    if top_10[0][2] == top_10[0][-1]:
      metrics['top_1'] += 1
    
    #top3
    if curr_query[2] in [i[-1] for i in top_10[:3]]:
      metrics['top_3'] += 1

    #top10
    if curr_query[2] in [i[-1] for i in top_10]:
      metrics['top_10'] += 1

  return {i : j/n_queries for i, j in metrics.items()}


metric_evaluation(all_queries, all_galleries, n_queries=30)
