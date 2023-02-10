import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle
import torchvision.transforms as transforms


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


def load_data():
      train = pd.read_csv('data/eye_dataset1/train.csv')
      test = pd.read_csv('data/eye_dataset1/test.csv')

      train_dir = os.path.join('data/eye_dataset1/train_images')
      test_dir = os.path.join('data/eye_dataset1/test_images')

      train['file_path'] = train['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
      test['file_path'] = test['id_code'].map(lambda x: os.path.join(test_dir,'{}.png'.format(x)))

      train['file_name'] = train["id_code"].apply(lambda x: x + ".png")
      test['file_name'] = test["id_code"].apply(lambda x: x + ".png")

      train['diagnosis'] = train['diagnosis'].astype(str)

      return train,test


from tqdm import tqdm

df_train,df_test = load_data()
print(df_train['diagnosis'].value_counts())
X_train=[]
Y_train=[]
count_0, count_1, count_2=0,0,0
print(type(df_train.diagnosis.iloc[0]))

for i in tqdm(range(0,len(df_train))):
 
  img = cv2.imread(df_train.file_path.iloc[i])

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

with open('data/x_train_eye_1_1', 'wb') as pickle_file:
    pickle.dump(X_train, pickle_file)
with open('data/y_train_eye_1_1', 'wb') as pickle_file:
    pickle.dump(Y_train, pickle_file)

