import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image

def MNIST(path, transform_train = None, transform_test = None):
  
    if transform_train is None:
        #TL
        print("TL transform Cifar")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229])
        ])
    if transform_test is None:
        #TL
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229])])

    train_dataset = torchvision.datasets.MNIST(root=path,
                                            train=True,
                                            transform=transform_train,download=True)

    test_dataset = torchvision.datasets.MNIST(root=path,
                                            train=False,
                                            transform=transform_test, download = True)
    return train_dataset, test_dataset


def CIFAR10(path, transform_train = None, transform_test = None):


    if transform_train is None:
        print("No Tl Cifar10")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform_train,download= True)

    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transform_test, download = True)
    return train_dataset, test_dataset

def CIFAR10_iid(num_clients, datapoints, path, transform_train, transform_test):


    #these transforms are same as Imagenet configurations as TL is being used

    if transform_train is None:
        #TL
        print("TL transform Cifar")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    if transform_test is None:
        #TL
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform_train,download= True)

    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transform_test, download = True)
    
    class2idx = train_dataset.class_to_idx.items()
    idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    new_train_dataset_size = num_clients * datapoints
    temp = len(train_dataset) - new_train_dataset_size

    print(len(train_dataset), new_train_dataset_size, temp)

    new_train_dataset,_ = torch.utils.data.random_split(train_dataset, (new_train_dataset_size, temp))
    new_test_dataset,_ = torch.utils.data.random_split(test_dataset, (2000, 8000))  #keeping 2k datapoints with each client

    return new_train_dataset, new_test_dataset, dict(class2idx), idx2class


def cifar10_setting2(num_clients, unique_datapoint, c_datapoints, path, transform_train, transform_test):

    if transform_train is None:
        #TL
        print("TL transform Cifar")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    if transform_test is None:
        #TL
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform_train,download= True)

    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transform_test, download = True)
    
    class2idx = train_dataset.class_to_idx.items()
    idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    #for setting two the datapoints are the equal datapoints 
    # so c1: 2000, c2 - c11: 150 each 
    new_train_dataset_size = unique_datapoint + ((num_clients - 1) * c_datapoints)
    temp = len(train_dataset) - new_train_dataset_size

    print(len(train_dataset), new_train_dataset_size, temp)

    new_train_dataset,_ = torch.utils.data.random_split(train_dataset, (new_train_dataset_size, temp))
    unique_train_dataset, common_train_dataset = torch.utils.data.random_split(new_train_dataset, (unique_datapoint, new_train_dataset_size - unique_datapoint))
    new_test_dataset,_ = torch.utils.data.random_split(test_dataset, (1000, 9000))  #keeping 2k datapoints with each client

    return unique_train_dataset, common_train_dataset, new_test_dataset, dict(class2idx), idx2class



def FashionMNIST(path, transform_train = None,transform_test = None):

    if transform_train is None:
        #TL
        print("TL transform Cifar")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229])
        ])
    if transform_test is None:
        #TL
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229])])


    train_dataset = torchvision.datasets.FashionMNIST(root=path,train=True,transform=transform_train,download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=path,train=False,transform=transform_test,download=True)

    return train_dataset, test_dataset


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['labels'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def get_duplicates(x):
    unique_list = list(df_undup['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'

def get_test_rows(x):
    # create a list of all the lesion_id's in the test set
    test_list = list(df_test['image_id'])
    if str(x) in test_list:
        return 'test'
    else:
        return 'train'

def HAM10000_data(path):
    data_dir='/home/manas/priv_SLR/data/HAM10000_data_files'
    images_dir = data_dir + "/images"
    all_image_path = glob(os.path.join(images_dir,'*.jpg'))
    print("gnc", len(all_image_path))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    #{"ISIC_0024306":/home/manas/priv_SLR/data/HAM10000_data_files/images/ISIC_0024306.jpg }

    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }
    
    df_original = pd.read_csv(data_dir + '/HAM10000_metadata')
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['labels'] = pd.Categorical(df_original['cell_type']).codes

    global df_undup
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    y = df_undup['labels']
    global df_test, df_train
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
    
    df_original['train_or_test'] = df_original['image_id']
    df_original['train_or_test'] = df_original['train_or_test'].apply(get_test_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_test'] == 'train']
   
    df_train=df_train[:8910]
    df_test=df_test[:1100]
  
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    
    norm_mean=[0.763038, 0.54564667, 0.57004464]
    norm_std = [0.14092727, 0.15261286, 0.1699712]
    train_transform = transforms.Compose([transforms.Resize((64, 64)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    test_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    train_dataset = HAM10000(df_train, transform=train_transform)
    test_dataset= HAM10000(df_test, transform=test_transform)

    return train_dataset, test_dataset



def load_full_dataset(dataset, dataset_path, num_clients, datapoints = None, pretrained = False,transform_train = None, transform_test = None):
    
    if dataset == 'mnist':
        train_dataset, test_dataset = MNIST(dataset_path)
        input_channels = 1
    
    if dataset == 'cifar10':


        if pretrained:
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

            transform_test =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

            train_dataset, test_dataset = CIFAR10(dataset_path, transform_train, transform_test)
        else:
            train_dataset, test_dataset = CIFAR10(dataset_path)

        input_channels = 3
        
        
    if dataset == 'ham10000':
        train_dataset, test_dataset = HAM10000_data(dataset_path)
        input_channels = 3   
         
    if dataset == "fmnist":
        train_dataset, test_dataset = FashionMNIST(dataset_path)
        input_channels = 1

    if dataset == "cifar10_tl":
        #this is the reduced train_dataset

        train_dataset, test_dataset, ci,ic = CIFAR10_iid(num_clients, datapoints, dataset_path, transform_train, transform_test)
        input_channels = 3

    if dataset == "cifar10_setting2":
        u_datapoints = 5000
        c_datapoints = 5000
        input_channels = 3
        u_train_dataset, c_train_dataset, test_dataset,_,_ = cifar10_setting2(num_clients, u_datapoints, c_datapoints, dataset_path, transform_train, transform_test)
        return u_train_dataset,c_train_dataset, test_dataset, input_channels
        
    return train_dataset, test_dataset, input_channels

    




# if __name__ == "__main__":
#     train_dataset, test_dataset = MNIST('../data')
#     print(type(train_dataset))
