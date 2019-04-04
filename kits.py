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
import os
from glob import glob
import sys
import random
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics import jaccard_similarity_score
import os
import nibabel as nib
import pickle
import cv2
from tqdm import tqdm, tnrange
from time import sleep



data_set = r"E:\kits19\data"    #data set path
   
all_patients =  next(os.walk(data_set))[1]      #get the name of the folder_names  


IMG_PIC_SIZE = 512    #to contol the size of each slice


slices_images_imageFile = []    #slices of the images that will be used for trainig
slices_images_segFile = []      #slices of the images that will be used for TARGET



tmp_save_dir = r"E:\temp"

max_arr=[]
min_arr = []
    

"""
Remarks:
insert tqdm for how many patients are done and how much slicing is done
take relevent images only from the nifiti images
Image data generator with saving all the pics into array in memory

extact ROI from seg file and crop that section only to inser into NN.
"""

def extract_cancer_slice(seg_file):
    
    seg_file = nib.load(seg_file).get_data()
    
    slices_having_kidney = []
    slices_having_cancer = []
    
    for slice_num in range(seg_file.shape[0]):
        slice_seg = seg_file[slice_num, :, :]
        
        max_val = slice_seg.max()
        if (max_val == 1):
            slices_having_kidney.append(slice_num)
        
        if (max_val == 2):
            slices_having_cancer.append(slice_num)
        
    return slices_having_cancer, slices_having_kidney    
        
      


for patient in tqdm(all_patients[:1], desc = "Outloop"):
    semi_full_path =  os.path.join(data_set, patient)

    files_per_patient =  next(os.walk(semi_full_path))[2]
    files_per_patient = sorted(files_per_patient, reverse =True)
    
    seg_file_path = os.path.join(semi_full_path, files_per_patient[0])
    cancer_slice_num , kidney_slice_num = extract_cancer_slice(seg_file_path)

    
    
    for file in tqdm(files_per_patient, desc = '2nd loop', leave=True):
        full_path =  os.path.join(semi_full_path, file)
        seg_or_image = nib.load(full_path).get_data()
        file_type, _, _= file.split(".")
        print(file_type)
        
        if (file_type == 'imaging'):
        #extract images slices from the 3D image
            for s in cancer_slice_num:
                cancer_slice = seg_or_image[s, :, :]
                slices_images_imageFile.append(cancer_slice)
        
        if (file_type == 'segmentation'):
        #extract slices from the 3D image
            for s in cancer_slice_num:
                cancer_slice = seg_or_image[s, :, :]
                slices_images_segFile.append(cancer_slice)
                
                   
            
            
        
        
                
            
        
Image_data = np.array(slices_images_imageFile)          #convert list to ndarray

Label_data = np.array(slices_images_segFile) 
                                                    #to check the size of traing and testing array
print("Label data : {:.2f}MB  \
      Image data  :{:.2f}MB ".format(Label_data.nbytes / (1024 * 1000.0), Image_data.nbytes / (1024 * 1000.0)))      
            
          




     
'''  
all the file has different channels like
(611, 512, 512)
(602, 512, 512)
(261, 512, 512)
(270, 512, 512)
(64, 512, 512)

'''
            

