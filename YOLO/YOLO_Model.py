#creating the path and producing the augmentation
import os
import cv2 as cv
import imutils
import numpy as np

####    FOLDERS SETUP

path = "C:/Users/matte/OneDrive/Desktop/MLProject/"

train_path = path + "training/"

test_path = path + "validation/"

animali = {}
animaliruotati = {}

####    FOLDER CREATION

for cartelle in os.listdir(train_path):
    try:
        os.mkdir(train_path + cartelle + "/rotated90")    
    except:
        print("Already created")
    try:
        os.mkdir(train_path + cartelle + "/rotated180")
    except:
        print("Already created")
    try:
        os.mkdir(train_path + cartelle + "/rotated270")
    except:
        print("Already created")
    try:
        os.mkdir(train_path + cartelle + "/verticalflip")
    except:
        print("Already created")
    try:
        os.mkdir(train_path + cartelle + "/horizontalflip")
    except:
        print("Already created")
    try:
        os.mkdir(train_path + cartelle + "/crop")
    except:
        print("Already created")
    

####    IMAGES DICTIONARY CREATION

for cartelle in os.listdir(train_path):
    animali[cartelle.split("(")[1][:-1]] = []
    animaliruotati[cartelle.split("(")[1][:-1]] = []
    
    animal_path = train_path + cartelle
    for lista_immagini in os.listdir(animal_path):
        if lista_immagini.endswith("JPEG"):
            animali[cartelle.split("(")[1][:-1]].append(animal_path + "/" + lista_immagini)
        if lista_immagini.endswith("JPEG"):
            animaliruotati[cartelle.split("(")[1][:-1]].append(animal_path + "/crop/" + lista_immagini)


####Func rotation
"""
def rotate_img(degree, diz):
    
#    --> return None.
#    --> accept a dictionary of files path
    
    for elem in folder:
        for pic in folder[elem]:
            if pic.endswith('JPEG'):
                #print(pic)
                load_img = cv.imread(pic)
                rot = imutils.rotate_bound(load_img, degree)
                rot_name = pic.rsplit('/', 1)
                cv.imwrite(rot_name[0] + f'/rotated{degree}' + rot_name[1], rot)
"""


####    90° ROTATION
"""
for animale in animali:
    for animal_pic in animali[animale]:
        if animal_pic.endswith("JPEG"):
            print(animal_pic)
            loaded_img = cv.imread(animal_pic)
            rotatedImage = imutils.rotate_bound(loaded_img, 90)
            #cv.imshow("rotatedImage", rotatedImage)
            #cv.waitKey(0)
            rotated_name = animal_pic.rsplit("/", 1)
            cv.imwrite(rotated_name[0] + "/rotated90/" + rotated_name[1], rotatedImage)
"""
####    180° ROTATION
"""
for animale in animali:
    for animal_pic in animali[animale]:
        if animal_pic.endswith("JPEG"):
            print(animal_pic)
            loaded_img = cv.imread(animal_pic)
            rotatedImage = imutils.rotate_bound(loaded_img, 180)
            #cv.imshow("rotatedImage", rotatedImage)
            #cv.waitKey(0)
            rotated_name = animal_pic.rsplit("/", 1)
            cv.imwrite(rotated_name[0] + "/rotated180/" + rotated_name[1], rotatedImage)
"""
####    270° ROTATION
"""
for animale in animali:
    for animal_pic in animali[animale]:
        if animal_pic.endswith("JPEG"):
            print(animal_pic)
            loaded_img = cv.imread(animal_pic)
            rotatedImage = imutils.rotate_bound(loaded_img, 270)
            #cv.imshow("rotatedImage", rotatedImage)
            #cv.waitKey(0)
            rotated_name = animal_pic.rsplit("/", 1)
            cv.imwrite(rotated_name[0] + "/rotated270/" + rotated_name[1], rotatedImage)
"""

print(animaliruotati)



import argparse
import random as rng
from matplotlib import pyplot as plt
import cv2 as cv
import time
rng.seed(12345)

###
best_contours = []
even_bett_lines = []
final_lines = []
last_line = []
com = 0
dilatation_size = 2
erosion_size = 5
img = cv.imread(animali["peacock"][73])

for this_img in animali["peacock"]:
    best_contours = []
    even_bett_lines = []
    final_lines = []
    last_line = []
    #print(this_img)
    img = cv.imread(this_img)

    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#    imggray = cv.bitwise_not(imggray)

    height = img.shape[0]
    width = img.shape[1]

    element_dil = cv.getStructuringElement(0, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    imggray = cv.dilate(imggray, element_dil)

    #plt.imshow(imggray)

    element_er = cv.getStructuringElement(0, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
        
    imggray = cv.erode(imggray, element_er)

    #plt.imshow(imggray)

    ret, thresh = cv.threshold(imggray, 100, 255, 0)
    plt.imshow(thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)



    for i in range(len(contours)):
        vert_min = 100000
        vert_max = 0
        oriz_min = 100000
        oriz_max = 0

        for pos in contours[i]:
            if pos[0][0] > oriz_max:
                oriz_max = pos[0][0]
            if pos[0][0] < oriz_min:
                oriz_min = pos[0][0]
            if pos[0][1] > vert_max:
                vert_max = pos[0][1]
            if pos[0][1] < vert_min:
                vert_min = pos[0][1]



        if not oriz_max - oriz_min < width/5 and not vert_max - vert_min < height/5:
            best_contours.append(contours[i])

        
    for lines in best_contours:
        vert_min = 100000
        vert_max = 0
        oriz_min = 100000
        oriz_max = 0
        
        for pos in lines:
            if pos[0][0] > oriz_max and width-pos[0][0]>20:
                oriz_max = pos[0][0]
            if pos[0][0] < oriz_min and pos[0][0]>20:
                oriz_min = pos[0][0]
            if pos[0][1] > vert_max and height-pos[0][1]>20:
                vert_max = pos[0][1]
            if pos[0][1] < vert_min and pos[0][1]>20:
                vert_min = pos[0][1]

        contour = np.array([(oriz_min, vert_min), (oriz_max, vert_min), (oriz_max, vert_max), (oriz_min, vert_max) ], dtype=np.int64)
        if not oriz_max - oriz_min < width/5 and not vert_max - vert_min < height/5:
            even_bett_lines.append(contour)

    for lin in range(0,len(even_bett_lines)):
        for el in range(0,len(even_bett_lines)):
            if lin != el:
                if even_bett_lines[lin][0][0]>=even_bett_lines[el][0][0] and even_bett_lines[lin][0][1]>=even_bett_lines[el][0][1] and even_bett_lines[lin][1][0]<=even_bett_lines[el][1][0] and even_bett_lines[lin][1][1]>=even_bett_lines[el][1][1] and even_bett_lines[lin][2][0]<=even_bett_lines[el][2][0] and even_bett_lines[lin][2][1]<=even_bett_lines[el][2][1] and even_bett_lines[lin][3][0]>=even_bett_lines[el][3][0] and even_bett_lines[lin][3][1]<=even_bett_lines[el][3][1]:
                    com += 1
        if com == 0:
            final_lines.append(even_bett_lines[lin])
        com = 0

    v_final = 100000
    b_final = 0
    l_final = 100000
    r_final = 0

    for lin in range(0, len(final_lines)):
        for linee in final_lines[lin]:
            if linee[0] < l_final:
                l_final = linee[0]
            if linee[0] > r_final:
                r_final = linee[0]
            if linee[1] < v_final:
                v_final = linee[1]
            if linee[1] > b_final:
                b_final = linee[1]
    if v_final != 100000 and l_final != 100000 and b_final != 0 and r_final != 0:
        last_line.append(np.array([(l_final, v_final), (r_final, v_final), (r_final, b_final), (l_final, b_final) ], dtype=np.int64))
    else:
        last_line.append(np.array([(0, 0), (width, 0), (width, height), (0, height) ], dtype=np.int64))

    #cv.drawContours(img, last_line, -1, (0,255,0), 2)

    #print(last_line[0][0][0],last_line[0][0][1], last_line[0][2][0],last_line[0][2][1])
    tlx = last_line[0][0][0]
    tly = last_line[0][0][1]
    brx = last_line[0][2][0]
    bry = last_line[0][2][1]
    #img = cv.imread(this_img)
    cropped_image = img[tly:bry, tlx:brx]
    cropped_name = this_img.rsplit("/", 1)
    cv.imwrite(cropped_name[0] + "/crop/" + cropped_name[1], cropped_image)
    #cv.imshow('image',img)
    #cv.waitKey(0)


test_img = "C:/Users/matte/OneDrive/Desktop/MLProject/training/n01644900(tailed_frog)/n01644900_27.JPEG"
test_img = "C:/Users/matte/OneDrive/Desktop/MLProject/training/n01806143(peacock)/n01806143_79.JPEG"
rows,cols,_ = img.shape
"""
for i in range(rows):
    for j in range(cols):
        k = img[i,j]
        print(k)"""

# grab the image channels, initialize the tuple of colors,
# the figure and the flattened feature vector
img = cv.imread(test_img)
chans = cv.split(img)
media_canali = {}
        

colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
for colore in colors:
    media_canali[colore] = []

# loop over the image channels
for this_img in animaliruotati["peacock"]:
    best_contours = []
    even_bett_lines = []
    final_lines = []
    last_line = []
    img = cv.imread(this_img)

    chans = cv.split(img)
    
    
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        
        media_canali[color].extend(chan)
        print(chan)

        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        # plot the histogram
        plt.plot(hist, color = color)
        plt.xlim([0, 256])

# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print("flattened feature vector size: %d" % (np.array(features).flatten().shape))
plt.show()
cv.waitKey(0)

#print(media_canali["b"])
print(len(media_canali["b"]))

for colore in media_canali.keys():
    print(len(media_canali[colore]))


final_top = {}

def ottieni_main_colors(img):
    this_img = cv.imread(img)
    top_colors = {}
    for linea in this_img:
        for pixel in linea:
            if pixel[0]//64 != pixel[1]//16 and pixel[1]//16 != pixel[2]//16:
                if "b_"+str(pixel[0]//16)+"_g_"+str(pixel[1]//16)+"_r_"+str(pixel[2]//16) in top_colors:
                    top_colors["b_"+str(pixel[0]//16)+"_g_"+str(pixel[1]//16)+"_r_"+str(pixel[2]//16)] += 1
                else:
                    top_colors["b_"+str(pixel[0]//16)+"_g_"+str(pixel[1]//16)+"_r_"+str(pixel[2]//16)] = 1
            else:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
    #print(top_colors)        
    #cv.imshow(img, this_img)
    #cv.waitKey()

    max_value = max(top_colors, key=top_colors.get)
    print(max_value)
    if "b_"+str(max_value.split("_")[1])+"_g_"+str(max_value.split("_")[3])+"_r_"+str(max_value.split("_")[5]) in final_top:
        final_top["b_"+str(max_value.split("_")[1])+"_g_"+str(max_value.split("_")[3])+"_r_"+str(max_value.split("_")[5])] += 1
    else:
        final_top["b_"+str(max_value.split("_")[1])+"_g_"+str(max_value.split("_")[3])+"_r_"+str(max_value.split("_")[5])] = 1        

for piccioni in animaliruotati["peacock"]:
    ottieni_main_colors(piccioni)

print(final_top)
