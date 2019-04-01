# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 06:44:42 2019

@author: tensor19

envirament name kids
"""

from starter_code.utils import load_case

volume, segmentation = load_case("00009")


from starter_code.visualize import visualize

visualize("case_00123", str("E:/temp/"))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import os
from glob import glob
import sys
import random
#from tqdm import tqdm_notebook
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.datasets as dsets
#from torch.autograd import Variable
#from tqdm import tqdm
import os
import nibabel as nib
import pickle





data_set = r"E:\kits19\data"


#get the name of the folder_names        
all_patients =  next(os.walk(data_set))[1]

imaging_arr = []
segmentation_arr = []

train=[]
target = []

for patient in all_patients[:5]:
    semi_full_path =  os.path.join(data_set,patient)
#    print(semi_full_path)
    files_per_patient =  next(os.walk(semi_full_path))[2]
    for file in files_per_patient:
        file_type, _, _= file.split(".")
        
        if file_type == "segmentation":
            full_path =  os.path.join(semi_full_path,file)
            print(full_path)
            img_seg = nib.load(full_path).get_data()
            print(img_seg.shape)
            target.append(img_seg)
#            segmentation_arr.append(full_path)
            
            
            
        if file_type == "imaging":
            full_path =  os.path.join(semi_full_path,file)
            print(full_path)
            img_kidney = nib.load(full_path).get_data()
            print(img_kidney.shape)
            train.append(img_kidney)
            imaging_arr.append(full_path)
            
#        print(full_path)
   
     
'''  
print("############################################")     

print(segmentation_arr) 


for seg in segmentation_arr:
    img_seg = nib.load(seg).get_data()
    print(img_seg.shape)


print(imaging_arr) 
        
for img in imaging_arr:
    img_seg = nib.load(img).get_data()
    print(img_seg.shape)
    


all the file has different channels like
(611, 512, 512)
(602, 512, 512)
(261, 512, 512)
(270, 512, 512)
(64, 512, 512)




for i in segmentation_arr:
    for i in img_seg[0]:
        
            '''
            

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split













