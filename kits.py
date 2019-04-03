# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 06:44:42 2019

@author: tensor19

envirament name kids
"""



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


#get the name of the folder_names        
all_patients =  next(os.walk(data_set))[1]

imaging_arr = []
segmentation_arr = []

IMG_PIC_SIZE = 512


train=[]
target = []
slices_images_imageFile = []
slices_images_segFile = []



tmp_save_dir = r"E:\temp"

max_arr=[]
min_arr = []
    

for patient in all_patients[:1]:
    semi_full_path =  os.path.join(data_set,patient)

    files_per_patient =  next(os.walk(semi_full_path))[2]
    for file in files_per_patient:
        file_type, _, _= file.split(".")
        
        if file_type == "segmentation":
            full_path =  os.path.join(semi_full_path,file)
            segmentation_arr.append(full_path)

            
            #segmentation file
            img_seg = nib.load(full_path).get_data()
            
            
            pic_having_kidey = []
            pic_having_cancer = []
            for slice_num_seg in range(img_seg.shape[0]):
#                count = count+1
                slice_seg = img_seg[slice_num_seg, :, :]
                max_val = slice_seg.max()
                if (max_val == 1):
                    pic_having_kidey.append(slice_num_seg)
                    
                if (max_val == 2):
                    pic_having_cancer.append(slice_num_seg)
                
                im_path = os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_seg)+".png"))
                plt.imsave(im_path,slice_seg , cmap='gray')

                slices_images_segFile.append(slice_seg)
                

            #append to the array
            target.append(img_seg)
            
            
            
        if file_type == "imaging":
            
            #append to the array
            imaging_arr.append(full_path)
        
            full_path =  os.path.join(semi_full_path,file)
            img_kidney = nib.load(full_path).get_data()

            
            train.append(img_kidney)

     
            for slice_num_kidney in range(img_kidney.shape[0]):
                slice_kidney = img_kidney[slice_num_kidney, :, :]
                im_path = os.path.join(tmp_save_dir , (patient+'_'+file_type+'_slice_'+str(slice_num_kidney)+".png"))
                plt.imsave(im_path , slice_kidney,cmap = 'gray')

                slices_images_imageFile.append(slice_kidney)
            
            
            
        
slices_images_imageFile_arr = np.array(slices_images_imageFile)

slices_images_segFile_arr = np.array(slices_images_segFile)
            
print("[INFO] data matrix: {:.2f}MB".format(slices_images_imageFile_arr.nbytes / (1024 * 1000.0)))  
            
print("[INFO] data matrix: {:.2f}MB".format(slices_images_segFile_arr.nbytes / (1024 * 1000.0)))      
            
          



  
     
'''  
all the file has different channels like
(611, 512, 512)
(602, 512, 512)
(261, 512, 512)
(270, 512, 512)
(64, 512, 512)

'''
            

