# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:29:57 2019

@author: Rochan Sharma, Data Scientist 

Kidney Tumor Segmentation Challenge, 2019 

All rights Reserved to "Reliance Industries Limited"

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



train_save_dir = r"E:\image_data\train"
test_save_dir = r"E:\image_data\test"


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
        


      
slices_images_imageFile_cancer = []    #slices of the images that will be used for trainig
slices_images_segFile_cancer = []      #slices of the images that will be used for TARGET


      
slices_images_imageFile_kidney = []    #slices of the images that will be used for trainig
slices_images_segFile_kidney = []      #slices of the images that will be used for TARGET



mean_Image_value = -68.59203445709038
std_Image_value = 0.40116888892385894


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
            seg_or_image = (seg_or_image - mean_Image_value) / std_Image_value
            seg_or_image = seg_or_image.astype(np.float32)
            
        #extract cancer images slices from the 3D image
            for s in cancer_slice_num:
                
                cancer_img_slice = seg_or_image[s, :, :]
                cancer_img_slice_height, cancer_img_slice_width = cancer_img_slice.shape[0], cancer_img_slice.shape[1]
                
                cancer_img_slice = cancer_img_slice[95:415 , 95:415]
                
#                im_path = os.path.join(train_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
#                plt.imsave(im_path, cancer_img_slice, cmap = 'gray')
                
                cancer_img_slice = np.expand_dims(cancer_img_slice, axis=-1)
           
                if (cancer_img_slice_height == 512 and cancer_img_slice_width == 512):
                        slices_images_imageFile_cancer.append(cancer_img_slice)
                else:
                    print("Dimention is not good", patient)
                    
                    
                    
        #extract kidney images slices from the 3D image
            for s in kidney_slice_num:
                
                kidney_img_slice = seg_or_image[s, :, :]
                kidney_img_slice_height, kidney_img_slice_width = kidney_img_slice.shape[0], kidney_img_slice.shape[1]
                
                kidney_img_slice = kidney_img_slice[95:415 , 95:415]
                
#                im_path = os.path.join(train_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
#                plt.imsave(im_path, cancer_img_slice, cmap = 'gray')
                
                kidney_img_slice = np.expand_dims(kidney_img_slice, axis=-1)
           
                if (kidney_img_slice_height == 512 and kidney_img_slice_width == 512):
                        slices_images_imageFile_kidney.append(kidney_img_slice)
                else:
                    print("Dimention is not good", patient)
                    
                    
                 
     
        if (file_type == 'segmentation'):
        #extract cancer slices from the 3D image
            for s in cancer_slice_num:
    
                cancer_slice_seg = seg_or_image[s, :, :]
                
                if cancer_slice_seg.max() == 1:
                    print("Max cancer_slice_seg vlaues is 1")
                
                cancer_slice_seg_height, cancer_slice_seg_width = cancer_slice_seg.shape[0], cancer_slice_seg.shape[1]
                cancer_slice_seg = cancer_slice_seg[95:415 , 95:415]
                
#                im_path = os.path.join(test_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
#                plt.imsave(im_path, cancer_slice_seg, cmap = 'gray')
                

                cancer_slice_seg = np.expand_dims(cancer_slice_seg, axis=-1)   #keras.json channel_last
               
                if (cancer_slice_seg_height == 512 and cancer_slice_seg_width == 512):
#                    continue
                    slices_images_segFile_cancer.append(cancer_slice_seg)
                else:
                    print("Dimention is not good for", patient)
                    
                    
                    
            for s in kidney_slice_num:
                kidney_slice_seg = seg_or_image[s, :, :]
                
                if kidney_slice_seg.max() == 2:
                    print("Max value in kidney_slice_seg is 2")
                
                kidney_slice_seg_height, kidney_slice_seg_width = kidney_slice_seg.shape[0], kidney_slice_seg.shape[1]
                kidney_slice_seg = kidney_slice_seg[95:415 , 95:415]
                
#                im_path = os.path.join(test_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
#                plt.imsave(im_path, kidney_slice_seg, cmap = 'gray')
                
                kidney_slice_seg = np.expand_dims(kidney_slice_seg, axis=-1)   #keras.json channel_last
               
                if (kidney_slice_seg_height == 512 and kidney_slice_seg_width == 512):
#                    continue
                    slices_images_segFile_kidney.append(kidney_slice_seg)
                else:
                    print("Dimention is not good for", patient)
                    
                    

#https://github.com/sharmarochan/KiTS/commit/39bcf23c190eb530512343c87c6cfb4b3ae387c2

import time
start = time.time()
      
        
Image_data_cancer = np.array(slices_images_imageFile_cancer)          #convert list to ndarray
#del slices_images_imageFile_cancer
Target_data_cancer = np.array(slices_images_segFile_cancer)

Image_data_kidney = np.array(slices_images_imageFile_kidney)          #convert list to ndarray
#del slices_images_imageFile_cancer
Target_data_kidney = np.array(slices_images_segFile_kidney)


end = time.time()
print(end - start)




print(Image_data_cancer.shape)      
print(Target_data_cancer.shape)  

print(Image_data_kidney.shape)      
print(Target_data_kidney.shape)  


# min max normalization
Image_data_cancer_std = (Image_data_cancer - Image_data_cancer.min()) / (Image_data_cancer.max() - Image_data_cancer.min())


Image_data_kidney_std = (Image_data_kidney - Image_data_kidney.min()) / (Image_data_kidney.max() - Image_data_kidney.min())



#img_flair = (img_flair - m_flair) / s_flair

#to check the size of traing and testing array
print("Image_data_cancer_std data : {:.2f}MB  \
      Target_data_cancer data  :{:.2f}MB ".format(Image_data_cancer_std.nbytes / (1024 * 1000.0), Target_data_cancer.nbytes / (1024 * 1000.0)))      



print("Image_data_kidney_std data : {:.2f}MB  \
      Target_data_kidney data  :{:.2f}MB ".format(Image_data_kidney_std.nbytes / (1024 * 1000.0), Target_data_kidney.nbytes / (1024 * 1000.0)))      






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
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf

X_train, X_valid, y_train, y_valid = train_test_split(Image_data_kidney_std, Target_data_kidney, test_size=0.20, random_state=2018)



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



im_width = 320
im_height = 320


##################################### Model Compile or RESET #####################################

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
#model.summary()


checkpoint_path = "E:\kits19\checkpoints_v2\cp-normalized-rms-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


callbacks = [
        EarlyStopping(patience=3, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
]


#callbacks = [
#        EarlyStopping(patience=10, verbose=1),
#        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
#]


#callbacks: class
#EarlyStopping: 5 number of epochs with no improvement after which training will be stopped
#ReduceLROnPlateau: reduce learning rate if no improvement is shown for 3 epochs
#ModelCheckpoint: Save the model after every epoch.

#Untrained model
#loss, acc = model.evaluate(X_valid, y_valid)
#print("Untrained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
#Untrained model, loss: 1187.48% , accuracy: 18.89%    on 5 patient data
#Untrained model, loss: 1027.48% , accuracy: 27.36%   for all patients


#Training of the model

results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


####### Runned upto here by Rochan #######

loss, acc = model.evaluate(X_valid, y_valid)
print("Trained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))

# Trained model, loss: 112.45% , accuracy: 70.74%   on 5 patient data

#Trained model, loss: 40.81% , accuracy: 97.76%  on sencond time training from 5th epoch to 10th epoch



"""
##########################     reset model ##########################
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb?hl=es-MX#scrollTo=IFPuhwntH8VH


model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
loss, acc = model.evaluate(X_valid, y_valid)
print("Rreset model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
#Reset model, loss: 1187.48% , accuracy: 18.89%    on 5 patient data
#Rreset model, loss: 1242.34% , accuracy: 18.93% on sencond time reset after 5th epoch
"""



plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


"""
The accuracy is very low and the error is very high without loading model weights
"""

########################## START : if the training is stopped or kernel is dead #######################


#latest = tf.train.latest_checkpoint(checkpoint_dir)
os.listdir(checkpoint_dir)


#initiate the model architecture of the model if the training is stopped or kernel is dead
latest = 'E:\kits19\checkpoints\cp-normalized-rms-0006.ckpt'
model.load_weights(latest)

# Restore model

loss, acc = model.evaluate(X_valid, y_valid)
print("Accuracy of model on Validation dataset, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
#Restored model, loss: 40.78% , accuracy: 97.98%




# Evaluate on validation set (this must be equals to the best log_loss)

model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["accuracy"])


# Load best model
#model.load_weights(checkpoint_path)
model.evaluate(X_valid, y_valid, verbose=1)


##########################  Inference with model  ##########################

# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)


Threshold = 0.5
# Threshold predictions
preds_train_t = (preds_train > Threshold).astype(np.uint8)
preds_val_t = (preds_val > Threshold).astype(np.uint8)



#write a function to loop over all the models and print outputs of same picture for each model.



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
    ax[1].set_title('Annotation')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Threshold Prediction'+str(Threshold));




plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)



count = 0
for i in range(len(X_train)):
    plot_sample(X_train, y_train, preds_train, preds_train_t, ix=i)
    count = count + 1
    
    if count == 10:
        break



count = 0
for i in range(len(X_train)):
    plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=200)
    count = count + 1
    
    if count == 10:
        break


######################################visualize results #######################

import matplotlib.pyplot as plt
import collections

def plot_images(original_img, ground_truth, predicted_img, threshold_img, mod):
    print("Inference of model ",mod)
    f,ax = plt.subplots(1,4,figsize=(20, 10))
    ax[0].imshow(original_img)#, cmap='gray')
    ax[0].set_title('Image')
    ax[1].imshow(ground_truth)#, cmap='gray')
    ax[1].set_title('Label')
    ax[2].imshow(predicted_img)#, cmap='gray')
    ax[2].set_title('Predicted')
    ax[3].imshow(threshold_img)#, cmap='gray')
    ax[3].set_title('Clean Img')
    plt.show()
    


for mod in range(1,9):
    
    model_path = "E:\kits19\checkpoints\cp-normalized-000"+str(mod)+".ckpt"
    print(model_path)
    print("Model is Reset")
    #model reset
    input_img = Input((im_height, im_width, 1), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["accuracy"])
#    loss, acc = model.evaluate(X_valid[:10], y_valid[:10])
#    print("Untrained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
    
    print("Model Loaded: ",mod )
    model.load_weights(model_path)

#    loss, acc = model.evaluate(X_valid[:10], y_valid[:10])
#    print("Trained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
    
    
    Threshold=0.5
    preds_val = model.predict(X_valid[:4], verbose=1)
    preds_val_t = (preds_val > Threshold).astype(np.uint8)
        
    for i in range(0,4):
        original_img = X_valid[i, ..., 0]
        ground_truth = y_valid[i].squeeze()
        predicted_img = preds_val[i].squeeze()
        threshold_img = preds_val_t[i].squeeze()
        plot_images(original_img, ground_truth, predicted_img, threshold_img, mod)
        
        
        map_vals = np.around(preds_val[i].flatten())
        count_ground_truth = collections.Counter(y_valid[i].flatten())
        count_predicted_img_round = collections.Counter(map_vals)
        count_predicted_img = collections.Counter(preds_val[i].flatten())
        sum(i > 1 for i in count_predicted_img)
        count_orignal_img = collections.Counter(original_img.flatten())
        
        count_1 = list(preds_val[i].flatten()).count(2)

        print("count_ground_truth is {}, count_predicted_img is {}".format(count_ground_truth, count_predicted_img))
        
  


###############################################################################

output_width = 320
output_height = 320

n_classes =3
colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

for imgName in range(len(X_valid)):
    img = X_valid[imgName]
    pr = model.predict( np.array([img]), verbose=1)[0]
    pr = pr.argmax( axis=2 )
    seg_img = np.zeros((output_height, output_width, 3))
    
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    
    seg_img = cv2.resize(seg_img  , (output_width, output_height ))
    cv2.imshow('image', seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break


#################### 0 -2  Masking #########################
    
latest = 'E:\kits19\checkpoints_v2\cp-normalized-adam_0_2_mask_model_test_run_secondTime-0041.ckpt'
model.load_weights(latest)

Target_data_cancer_0_2_mask = np.where(Target_data_cancer==1, 0, Target_data_cancer)


X_train, X_valid, y_train, y_valid = train_test_split(Image_data_cancer_std, Target_data_cancer_0_2_mask, test_size=0.20, random_state=2018)



##################################### Model Compile or RESET #####################################

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
#model.summary()


checkpoint_path = "E:\kits19\checkpoints_v2\cp-normalized-adam_0_2_mask_model_test_run_secondTime-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


#callbacks = [
#        EarlyStopping(patience=3, verbose=1),
#        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
#]


callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
]


#callbacks: class
#EarlyStopping: 5 number of epochs with no improvement after which training will be stopped
#ReduceLROnPlateau: reduce learning rate if no improvement is shown for 3 epochs
#ModelCheckpoint: Save the model after every epoch.

#Untrained model
#loss, acc = model.evaluate(X_valid, y_valid)
#print("Untrained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
#Untrained model, loss: 1187.48% , accuracy: 18.89%    on 5 patient data
#Untrained model, loss: 1027.48% , accuracy: 27.36%   for all patients


#Training of the cancer model

results_0_2_mask = model.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


####### Runned upto here by Rochan #######

loss_0_2_mask, acc_0_2_mask = model.evaluate(X_valid, y_valid)
print("Trained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss_0_2_mask, 100*acc_0_2_mask))



plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results_0_2_mask.history["loss"], label="loss")
plt.plot(results_0_2_mask.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results_0_2_mask.history["val_loss"]), np.min(results_0_2_mask.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


#best model is model 41 for 0_2 mask


preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)


Threshold = 0.5
# Threshold predictions
preds_train_t = (preds_train > Threshold).astype(np.uint8)
preds_val_t = (preds_val > Threshold).astype(np.uint8)




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
    ax[1].set_title('Annotation')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Threshold Prediction'+str(Threshold));




plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)










# Trained model, loss: 112.45% , accuracy: 70.74%   on 5 patient data

#Trained model, loss: 40.81% , accuracy: 97.76%  on sencond time training from 5th epoch to 10th epoch

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results_0_2_mask.history["loss"], label="loss")
plt.plot(results_0_2_mask.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results_0_2_mask.history["val_loss"]), np.min(results_0_2_mask.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


















#################### 0 -1  Masking #########################
    
latest = 'E:\kits19\checkpoints_v2\cp-normalized-adam_0_1_mask_model_test_run_secondTime-0037.ckpt'
model.load_weights(latest)

Target_data_cancer_0_1_mask = np.where(Target_data_cancer==2, 0, Target_data_cancer)


X_train, X_valid, y_train, y_valid = train_test_split(Image_data_cancer_std, Target_data_cancer_0_1_mask, test_size=0.20, random_state=2018)



##################################### Model Compile or RESET #####################################

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
#model.summary()


checkpoint_path = "E:\kits19\checkpoints_v2\cp-normalized-adam_0_1_mask_model_test_run_secondTime-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


#callbacks = [
#        EarlyStopping(patience=3, verbose=1),
#        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
#]


callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period = 1)
]


#callbacks: class
#EarlyStopping: 5 number of epochs with no improvement after which training will be stopped
#ReduceLROnPlateau: reduce learning rate if no improvement is shown for 3 epochs
#ModelCheckpoint: Save the model after every epoch.

#Untrained model
#loss, acc = model.evaluate(X_valid, y_valid)
#print("Untrained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss, 100*acc))
#Untrained model, loss: 1187.48% , accuracy: 18.89%    on 5 patient data
#Untrained model, loss: 1027.48% , accuracy: 27.36%   for all patients


#Training of the cancer model

results_0_1_mask = model.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))


####### Runned upto here by Rochan #######

loss_0_1_mask, acc_0_1_mask = model.evaluate(X_valid, y_valid)
print("Trained model, loss: {:5.2f}% , accuracy: {:5.2f}%".format(100*loss_0_1_mask, 100*acc_0_1_mask))



plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results_0_1_mask.history["loss"], label="loss")
plt.plot(results_0_1_mask.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results_0_1_mask.history["val_loss"]), np.min(results_0_1_mask.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_0_1_mask_loss")
plt.legend();


#best model is model 37  for 0_1 mask


preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)


Threshold = 0.5
# Threshold predictions
preds_train_t = (preds_train > Threshold).astype(np.uint8)
preds_val_t = (preds_val > Threshold).astype(np.uint8)




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
    ax[1].set_title('Annotation')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Prediction')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Threshold Prediction'+str(Threshold));




plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)










# Trained model, loss: 112.45% , accuracy: 70.74%   on 5 patient data

#Trained model, loss: 40.81% , accuracy: 97.76%  on sencond time training from 5th epoch to 10th epoch

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results_0_1_mask.history["loss"], label="loss")
plt.plot(results_0_1_mask.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results_0_1_mask.history["val_loss"]), np.min(results_0_1_mask.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
