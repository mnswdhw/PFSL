import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Necessary utility modules and libraries
import os
import shutil
import pathlib
import random
import datetime
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from scipy.signal import gaussian, convolve2d
# import seaborn as sns

# Libraries for building the model
# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout, Activation, GlobalAveragePooling2D, BatchNormalization, GlobalMaxPooling2D
# from tensorflow.keras.applications import DenseNet121, ResNet50, InceptionV3, Xception, VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import backend
# from tensorflow.keras.regularizers import l2, l1
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm

dir_path='data/eye_dataset2/eyepacs_preprocess'
df_temp = pd.read_csv("data/eye_dataset2/train_eyepacs.csv")
print(len(df_temp))
df_temp['level'].value_counts()


def load_data():
      train = pd.read_csv("data/eye_dataset2/train_eyepacs.csv")
    #   test = pd.read_csv('data/test.csv')

      train_dir = os.path.join('data/eye_dataset2/eyepacs_preprocess')
    #   test_dir = os.path.join('data/test_images')

      train['file_path'] = train['image'].map(lambda x: os.path.join(train_dir,'{}.jpeg'.format(x)))
    #   test['file_path'] = test['id_code'].map(lambda x: os.path.join(test_dir,'{}.png'.format(x)))

      train['file_name'] = train["image"].apply(lambda x: x + ".jpeg")
    #   test['file_name'] = test["id_code"].apply(lambda x: x + ".png")

      train['diagnosis'] = train['level'].astype(str)

      return train


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def isbright(image, dim=227, thresh=0.4):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel 
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
    
def circle_crop(img, sigmaX):   
    """
    Create circular crop around image centre    
    """ 
       
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img) 
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.resize(img, (224, 224))
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
   
    return img 



def image_preprocessing(img):
    # 1. Read the image
    #     img = mpimg.imread(img_path)
    img = img.astype(np.uint8)
    
    # 2. Extract the green channel of the image
    b, g, r = cv2.split(img)
    
    # 3.1. Apply CLAHE to intensify the green channel extracted image
    clh = cv2.createCLAHE(clipLimit=4.0)
    g = clh.apply(g)
    
    # 3.2. Convert enhanced image to grayscale
    merged_bgr_green_fused = cv2.merge((b, g, r))
    img_bw = cv2.cvtColor(merged_bgr_green_fused, cv2.COLOR_BGR2GRAY)
    
    # 4. Remove the isolated pixels using morphological cleaning operation.
    kernel1 = np.ones((1, 1), np.uint8)
    morph_open = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel1)
    
    # 5. Extract blood vessels using mean-C thresholding.
    thresh = cv2.adaptiveThreshold(morph_open, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    kernel2 = np.ones((2, 2), np.uint8)
    morph_open2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2)
    
    # 6. Stacking the image into 3 channels
    stacked_img = np.stack((morph_open2,)*3, axis=-1)
    
    return stacked_img.astype("float64")


df_train=load_data()
print(df_train['diagnosis'].value_counts())
X_train=[]
Y_train=[]
count_0, count_1, count_2=0,0,0
print(type(df_train.diagnosis.iloc[0]))

idxs=list(np.random.choice( list(range(0,35000)),5000, replace = False))


for idx in tqdm(range(0,len(idxs))):
  i=idxs[idx]
  img = cv2.imread(df_train.file_path.iloc[i])
#   print(type(img))
 
  img = circle_crop(img,sigmaX=10)
  X_train.append(img)

  temp=[]

  if(int(df_train.diagnosis.iloc[i])==0):
    count_0+=1
    ans=0
  elif(int(df_train.diagnosis.iloc[i])==1 or int(df_train.diagnosis.iloc[i])==2):
    count_1+=1
    ans=1
  elif(int(df_train.diagnosis.iloc[i])==3 or int(df_train.diagnosis.iloc[i])==4):
    count_2+=1
    ans=2
  Y_train.append(ans)

with open('data/x_train_eye_2', 'wb') as pickle_file:
    pickle.dump(X_train, pickle_file)
with open('data/y_train_eye_2', 'wb') as pickle_file:
    pickle.dump(Y_train, pickle_file)

print(count_0)
print(count_1)
print(count_2)

