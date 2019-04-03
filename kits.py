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
import matplotlib
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
import cv2



data_set = r"E:\kits19\data"


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice)   



#get the name of the folder_names        
all_patients =  next(os.walk(data_set))[1]

imaging_arr = []
segmentation_arr = []

IMG_PIC_SIZE = 512


train=[]
target = []
slices_images_imageFile = []
slices_images_segFile = []


'''

def check_image_black_or_white(img):
    extrema = img.convert("L").getextrema()
    if extrema == (0, 0):  # all black
        print("black")
    elif extrema == (1, 1):
        print("white")
        # all white
#    print(extrema)
'''


tmp_save_dir = r"E:\temp"

max_arr=[]
min_arr = []

def is_sorta_black(arr, threshold=0.7):
    tot = np.float(np.sum(arr))
    if tot/arr.size  > (1-threshold):
       print ("is not black")
#       return False
    else:
       print ("is kinda black")
#       return True
   
    
    

for patient in all_patients[:1]:
    semi_full_path =  os.path.join(data_set,patient)
#    print(semi_full_path)
    files_per_patient =  next(os.walk(semi_full_path))[2]
    for file in files_per_patient:
        file_type, _, _= file.split(".")
        
        if file_type == "segmentation":
            full_path =  os.path.join(semi_full_path,file)
            segmentation_arr.append(full_path)
            
#            print(full_path)
            
            #segmentation file
            img_seg = nib.load(full_path).get_data()
            print(img_seg.shape)
            
            
            #plot the slices of the image
            slice_0 = img_seg[95, :, :]
            slice_1 = img_seg[201, :, :]
            slice_2 = img_seg[300, :, :]
            
            print(type(slice_1))
            print(np.argmax(slice_2))
            print(type(slice_1[162,480]))
            
            show_slices([slice_0, slice_1, slice_2])
            plt.suptitle("Center slices for segmentation image")
            
#            count = 0
            pic_having_kidey = []
            pic_having_cancer = []
            for slice_num_seg in range(img_seg.shape[0]):
#                count = count+1
                slice_seg = img_seg[slice_num_seg, :, :]
                max_val = slice_seg.max()
                if (max_val == 1):
                    print(max_val)
                    pic_having_kidey.append(slice_num_seg)
                    
                if (max_val == 2):
                    pic_having_cancer.append(slice_num_seg)
                    
                    
                print(type(slice_seg))
                print(type(slice_seg[0,0]))
                max_arr.append(np.max(slice_seg))
                min_arr.append(np.min(slice_seg))
                
                im_path = os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_seg)+".png"))
#                plt.imsave(im_path,slice_seg , cmap='gray')
                
                
                
#                slice_seg = cv2.resize(slice_seg,(IMG_PIC_SIZE , IMG_PIC_SIZE))
#                cv2.imwrite(os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_seg)+".png")),slice_seg)
                is_sorta_black(slice_seg)
                slices_images_segFile.append(slice_seg)
                
                
                
                
            #append to the array
            target.append(img_seg)
            
            
            
        if file_type == "imaging":
            
            #append to the array
            imaging_arr.append(full_path)
            
            full_path =  os.path.join(semi_full_path,file)
            print(full_path)
            img_kidney = nib.load(full_path).get_data()
#            print(img_kidney.shape)
            
            train.append(img_kidney)
            
            #plot the slices of the image
            slice_0 = img_kidney[95, :, :]
            slice_1 = img_kidney[201, :, :]
            slice_2 = img_kidney[300, :, :]
            show_slices([slice_0, slice_1, slice_2])
            plt.suptitle("Center slices for imaging image")
            
            
            
            for slice_num_kidney in range(img_kidney.shape[0]):
                slice_kidney = img_kidney[slice_num_kidney, :, :]
                im_path = os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_kidney)+".png"))
#                plt.imsave(im_path,slice_kidney,cmap = 'gray')
                
                
#                slice_kidney = cv2.resize(slice_kidney,(IMG_PIC_SIZE,IMG_PIC_SIZE))
#                cv2.imwrite(os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_kidney)+".png")),slice_kidney)
#                print(slice_kidney.shape)
                
                
                is_sorta_black(slice_kidney)
                slices_images_imageFile.append(slice_kidney)
                
        

            
#        print(full_path)
            
            
            
        
slices_images_imageFile_arr = np.array(slices_images_imageFile)

slices_images_segFile_arr = np.array(slices_images_segFile)
            
print("[INFO] data matrix: {:.2f}MB".format(slices_images_imageFile_arr.nbytes / (1024 * 1000.0)))  
            
print("[INFO] data matrix: {:.2f}MB".format(slices_images_segFile_arr.nbytes / (1024 * 1000.0)))      
            
          



  
     
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













