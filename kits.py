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

for patient in all_patients[:1]:
    semi_full_path =  os.path.join(data_set,patient)

    files_per_patient =  next(os.walk(semi_full_path))[2]
    for file in files_per_patient:
        file_type, _, _= file.split(".")
        full_path =  os.path.join(semi_full_path,file)
        
        if file_type == "segmentation":     #segmentation file
            
            img_seg = nib.load(full_path).get_data()   #load nifti file as ndarrray
            
            
            pic_having_kidey = []           #slices having kidney not cancer
            pic_having_cancer = []          #slices having kidney and cancer section
            
            
            for slice_num_seg in range(img_seg.shape[0]):
                slice_seg = img_seg[slice_num_seg, :, :]        #slices the 3d image based on depth
                
                max_val = slice_seg.max()                       #extract the max value based on the label of the pixels
                if (max_val == 1):                              # detecting pic having kidney only
                    pic_having_kidey.append(slice_num_seg)
                    
                if (max_val == 2):                              # detecting pic having cancer and kideny section
                    pic_having_cancer.append(slice_num_seg)
                
                im_path = os.path.join(tmp_save_dir,(patient+'_'+file_type+'_slice_'+str(slice_num_seg)+".png"))
                plt.imsave(im_path,slice_seg , cmap='gray')     #save image to directroy

                slices_images_segFile.append(slice_seg)         #append to targrt array
        
        
        if file_type == "imaging":                              #for full MRI image
            img_kidney = nib.load(full_path).get_data()         #get nd array from nifti

     
            for slice_num_kidney in range(img_kidney.shape[0]):
                slice_kidney = img_kidney[slice_num_kidney, :, :]
                im_path = os.path.join(tmp_save_dir , (patient+'_'+file_type+'_slice_'+str(slice_num_kidney)+".png"))
                plt.imsave(im_path , slice_kidney,cmap = 'gray')        #save to directory

                slices_images_imageFile.append(slice_kidney)   #for training 
            
            
            
        
Image_data = np.array(slices_images_imageFile)          #convert list to ndarray

Label_data = np.array(slices_images_segFile) 
                                                    #to check the size of traing and testing array
print("Label data : {:.2f}MB  \
      Image data  :{:.2f}MB ".format(Label_data.nbytes / (1024 * 1000.0) , Image_data.nbytes / (1024 * 1000.0)))      
            
          



  
     
'''  
all the file has different channels like
(611, 512, 512)
(602, 512, 512)
(261, 512, 512)
(270, 512, 512)
(64, 512, 512)

'''
            

