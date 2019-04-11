# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 06:44:42 2019

@author: tensor19

Kidney Tumor Segmentation Challenge, 2019 

All rights Reserved to RIL

envirament name kids
chainer
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
import png


data_set = r"E:\kits19\data"    #data set path
   
all_patients =  next(os.walk(data_set))[1]      #get the name of the folder_names  


IMG_PIC_SIZE = 300    #to contol the size of each slice




train_save_dir = r"E:\image_data\train"
test_save_dir = r"E:\image_data\test"

max_arr=[]
min_arr = []
    

"""
Remarks:
    Done:
insert tqdm for how many patients are done and how much slicing is done
take relevent images only from the nifiti images

    Pending:
Image data generator with saving all the pics into array in memory
Make a CSV file for storing the position and depth of the images.
extact ROI from seg file and crop that section only to inser into NN.
*********
Train on Half dataset for longer epoches and see results. Post processing
Reduce the size of the image
*********

convert ndarray to nifti


    Note:
all the file has different channels like
(611, 512, 512)
(602, 512, 512)
(261, 512, 512)
(270, 512, 512)
(64, 512, 512)


patient 160's dimention is not same


Numpy Reshape doesn't change the data

"""

def extract_cancer_slice(seg_file):
    
    seg_file = nib.load(seg_file).get_data()
    
    slices_having_kidney = []
    slices_having_cancer = []
#    slices_having_both_cancer_and_kidney = []
    
    for slice_num in range(seg_file.shape[0]):
        slice_seg = seg_file[slice_num, :, :]
        
        max_val = slice_seg.max()
        if (max_val == 1):
            slices_having_kidney.append(slice_num)
        
        if (max_val == 2):
            slices_having_cancer.append(slice_num)
            
        ''' 
        if (max_val == 1 or max_val ==2):
            slices_having_both_cancer_and_kidney.append(slice_num)
    print('slices_having_both_cancer_and_kidney' , slices_having_both_cancer_and_kidney)
    print(len(slices_having_both_cancer_and_kidney))
        '''
        
    return slices_having_cancer, slices_having_kidney    
        


      
slices_images_imageFile = []    #slices of the images that will be used for trainig
slices_images_segFile = []      #slices of the images that will be used for TARGET

print("Loading dataset...")
#slices_images_seg = np.empty()
for patient in tqdm(all_patients):
    semi_full_path =  os.path.join(data_set, patient)

    files_per_patient =  next(os.walk(semi_full_path))[2]
    files_per_patient = sorted(files_per_patient, reverse =True)
    
    seg_file_path = os.path.join(semi_full_path, files_per_patient[0])
    cancer_slice_num , kidney_slice_num = extract_cancer_slice(seg_file_path)

    
    
    for file in files_per_patient:
        full_path =  os.path.join(semi_full_path, file)
        seg_or_image = nib.load(full_path).get_data()
        img_data_arr = np.asarray(seg_or_image)
        file_type, _, _= file.split(".")
        
        
        
        if (file_type == 'imaging'):
        #extract images slices from the 3D image
            for s in cancer_slice_num:
#                im_path = os.path.join(train_save_dir,(patient+'_'+file_type+'_slice_'+str(s)+".png"))
                
                cancer_img_slice = seg_or_image[s, :, :]
                cancer_img_slice_height, cancer_img_slice_width = cancer_img_slice.shape[0], cancer_img_slice.shape[1]
                
                cancer_img_slice = cancer_img_slice[100:400 , 100:400]
                cancer_img_slice = np.expand_dims(cancer_img_slice, axis=-1)
#                cancer_img_slice = np.expand_dims(cancer_img_slice, axis=0)
                #slices_images_imageFile = np.append(slices_images_imageFile, cancer_img_slice, axis=0)
                
                if (cancer_img_slice_height == 512 and cancer_img_slice_width ==512):
#                    continue
                        slices_images_imageFile.append(cancer_img_slice)
                else:
                    print("Dimention is not good", patient)
#                plt.imsave(im_path, cancer_img_slice , cmap='gray')
            
            
#                if s==290:
#                    plt.imshow(cancer_img_slice)
     
        if (file_type == 'segmentation'):
        #extract slices from the 3D image
            for s in cancer_slice_num:
#                im_path = os.path.join(test_save_dir,(patient+'_'+file_type+'_slice_'+str(s)+".png"))
                
                cancer_slice_seg = seg_or_image[s, :, :]
                cancer_slice_seg_height, cancer_slice_seg_width = cancer_slice_seg.shape[0], cancer_slice_seg.shape[1]
                
                cancer_slice_seg = cancer_slice_seg[100:400 , 100:400]
                cancer_slice_seg = np.expand_dims(cancer_slice_seg, axis=-1)   #keras.json channel_last
#                cancer_slice_seg = np.expand_dims(cancer_slice_seg, axis=0)
                
                if (cancer_slice_seg_height == 512 and cancer_slice_seg_width ==512):
#                    continue
                    slices_images_segFile.append(cancer_slice_seg)
                else:
                    print("Dimention is not good", patient)

                #np.append(slices_images_segFile, cancer_slice_seg, axis=0)
#                plt.imsave(im_path, cancer_slice_seg , cmap='gray')
                
#                if s==290:
#                    plt.imshow(cancer_slice_seg)
      



import time
start = time.time()
      
        
Image_data = np.array(slices_images_imageFile)          #convert list to ndarray
#del slices_images_imageFile
Target_data = np.array(slices_images_segFile)


end = time.time()
print(end - start)



'''
ValueError: could not broadcast input array from shape (512,512,1) into shape (512)

4772 samples for training and 843 samples for testing  (0.15)

4492 samples for training and 1123 samples for testing  (0.20)

'''

Image_data.shape
Target_data.shape


#Image_data = np.expand_dims(Image_data, axis=-1)
#Target_data = np.expand_dims(Label_data, axis=-1)


#to check the size of traing and testing array
print("Label data : {:.2f}MB  \
      Image data  :{:.2f}MB ".format(Target_data.nbytes / (1024 * 1000.0), Image_data.nbytes / (1024 * 1000.0)))      


print(Image_data.shape)      
print(Target_data.shape)   




####################UNET#########################
#https://www.depends-on-the-definition.com/unet-keras-segmenting-images/


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


X_train, X_valid, y_train, y_valid = train_test_split(Image_data, Target_data, test_size=0.20, random_state=2018)



def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model



im_width = 512
im_height = 512




input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('kits_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

#callbacks: class
#EarlyStopping: 5 number of epochs with no improvement after which training will be stopped
#ReduceLROnPlateau: reduce learning rate if no improvement is shown for 3 epochs
#ModelCheckpoint: Save the model after every epoch.


results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))





plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();




# Load best model
model.load_weights('kits_model.h5')



# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)


# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)



def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Image')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('cancer')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Image')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Cancer Predicted');




plot_sample(X_train, y_train, preds_train, preds_train_t, ix=200)


plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=200)
    





