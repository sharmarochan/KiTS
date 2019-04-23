# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 06:21:42 2019

@author: tensor19
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
        


      
slices_images_imageFile = []    #slices of the images that will be used for trainig
slices_images_segFile = []      #slices of the images that will be used for TARGET


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
        
        #for header information
        seg_or_image_load = nib.load(full_path)
        n1_header = seg_or_image_load.header
#        print(n1_header)
        
        img_data_arr = np.asarray(seg_or_image)
        file_type, _, _= file.split(".")
        
        
        
        if (file_type == 'imaging'):
            seg_or_image = (seg_or_image - mean_Image_value) / std_Image_value
            seg_or_image = seg_or_image.astype(np.float32)
            
        #extract images slices from the 3D image
            for s in cancer_slice_num:
                
                cancer_img_slice = seg_or_image[s, :, :]
                cancer_img_slice_height, cancer_img_slice_width = cancer_img_slice.shape[0], cancer_img_slice.shape[1]
                
                cancer_img_slice = cancer_img_slice[95:415 , 95:415]
                
                im_path = os.path.join(train_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
                plt.imsave(im_path, cancer_img_slice, cmap = 'gray')
                
                cancer_img_slice = np.expand_dims(cancer_img_slice, axis=-1)
           
                if (cancer_img_slice_height == 512 and cancer_img_slice_width ==512):
                        slices_images_imageFile.append(cancer_img_slice)
                else:
                    print("Dimention is not good", patient)

     
        if (file_type == 'segmentation'):
        #extract slices from the 3D image
            for s in cancer_slice_num:
    
                cancer_slice_seg = seg_or_image[s, :, :]
                
                if cancer_slice_seg.max() == 1:
                    print("Max cancer_slice_seg vlaues is 1")
                
                cancer_slice_seg_height, cancer_slice_seg_width = cancer_slice_seg.shape[0], cancer_slice_seg.shape[1]
                
                cancer_slice_seg = cancer_slice_seg[95:415 , 95:415]
                
                im_path = os.path.join(test_save_dir, (patient+'_'+file_type+'_slice_'+str(s)+".png"))
                plt.imsave(im_path, cancer_slice_seg, cmap = 'gray')
                
                
                cancer_slice_seg = np.expand_dims(cancer_slice_seg, axis=-1)   #keras.json channel_last
               
                if (cancer_slice_seg_height == 512 and cancer_slice_seg_width ==512):
#                    continue
                    slices_images_segFile.append(cancer_slice_seg)
                else:
                    print("Dimention is not good for", patient)
                    
 
    

import time
start = time.time()
      
        
Image_data = np.array(slices_images_imageFile)          #convert list to ndarray
#del slices_images_imageFile
Target_data = np.array(slices_images_segFile)


end = time.time()
print(end - start)



print(Image_data.shape)      
print(Target_data.shape)   

# min max normalization
Image_data_std = (Image_data - Image_data.min()) / (Image_data.max() - Image_data.min())





#img_flair = (img_flair - m_flair) / s_flair

#to check the size of traing and testing array
print("Label data : {:.2f}MB  \
      Image data  :{:.2f}MB ".format(Target_data.nbytes / (1024 * 1000.0), Image_data_std.nbytes / (1024 * 1000.0)))      






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join
import glob
import sys
import random
import warnings
from tqdm import tqdm
import itertools
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import initializers, layers, models
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks


# Remember to enable GPU






#we already have mask

def openCVdemo():
    ID = 'case_00000_imaging_slice_288'
    FILE = r"E:\image_data\train/{}.png".format(ID)
    img = cv2.imread(FILE,0)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    # Plot Here
    plt.figure(figsize=(15,5))
    images = [blur, 0, th3]
    titles = ['Original Image (X_train)','Gaussian filtered Image (OpenCV)',"Segmentated Image (OpenCV)"]
    plt.subplot(1,3,1),plt.imshow(img,'gray')
    plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(images[0],'gray')
    plt.title(titles[1]), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(images[2],'gray')
    plt.title(titles[2]), plt.xticks([]), plt.yticks([])
openCVdemo()




# Set some parameters
IMG_WIDTH = 320
IMG_HEIGHT = 320
IMG_CHANNELS = 3
TRAIN_PATH = r'E:\image_data\train'
TEST_PATH = r'E:\image_data\test'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))
test_ids = next(os.walk(TEST_PATH))

# Get and resize train images and masks
X_train = np.zeros((len(train_ids[2]), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids[2]), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

train_count =0
for id_ in next(os.walk(TRAIN_PATH))[2]:
    path = TRAIN_PATH +'/'+str(id_)
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[train_count] = img
    
    train_count = train_count + 1
    if train_count % 100==0:
        print("train_count",train_count)
  
    
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float64) 
test_count = 0
for mask_file in next(os.walk(TEST_PATH))[2]:
    mask_ = imread(TEST_PATH+'/'+mask_file)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                  preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)
    Y_train[test_count] = mask
    test_count = test_count + 1
    if test_count % 100==0:
        print("test_count",test_count)




'''
# Get and resize test images
X_test = np.zeros((len(test_ids[2]), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids[2]), total=len(test_ids[2])):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
print('Done!')
'''


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2)
print('\nx_train',x_train.shape)
print('x_test',x_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)



print("Label data : {:.2f}MB  \
      Image data  :{:.2f}MB ".format(X_train.nbytes / (1024 * 1000.0), Y_train.nbytes / (1024 * 1000.0)))      











class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

#RLE encoding for submission
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
    


def SIMPLE(a,b,c,d):
    smooth = 1.
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    simple_cnn = Sequential()
    simple_cnn.add(BatchNormalization(input_shape = (None, None, IMG_CHANNELS),name = 'NormalizeInput'))
    simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
    simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))
    # use dilations to get a slightly larger field of view
    simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
    simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))
    simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))
    # the final processing
    simple_cnn.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))
    simple_cnn.add(Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))
    simple_cnn.summary()
    
    checkpoint_path = "E:\kits19\checkpoints\simple-conv-{epoch:04d}.ckpt"
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, period = 1)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    simple_cnn.compile(optimizer = 'adam', 
                       loss = "mean_squared_error", 
                       metrics = ["accuracy"])
    
    
    history = simple_cnn.fit(x_train,y_train, validation_data=(x_test,y_test),callbacks = [earlystopper, checkpointer, MetricsCheckpoint('logs')], epochs = 100)
    plot_learning_curve(history)
    plt.show()
    plotKerasLearningCurve()
    plt.show()
    global modelY
    modelY = simple_cnn
    return modelY


SIMPLE(x_train, y_train,x_test,y_test)
    
    

