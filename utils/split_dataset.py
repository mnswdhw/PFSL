import os
import torch
import sys
from utils import datasets
import pickle
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def split_dataset(dataset: str, client_ids: list, datapoints = None, pretrained = False, output_dir='data' ):
    print('Splitting dataset (may take some time)...', end='')

    num_clients = len(client_ids)
    
    train_dataset, test_dataset, input_channels = datasets.load_full_dataset(dataset, output_dir,num_clients, datapoints, pretrained)
    per_client_trainset_size = len(train_dataset)//len(client_ids)
    train_split = [per_client_trainset_size]*len(client_ids)
    # train_split.append(len(train_dataset)-per_client_trainset_size*(len(client_ids)-1))

    per_client_testset_size = len(test_dataset)//len(client_ids)
    test_split = [per_client_testset_size]*len(client_ids)
    # test_split.append(len(test_dataset)-test_batch_size*(num_clients-1))

    train_datasets = list(torch.utils.data.random_split(train_dataset, train_split))
    test_datasets = list(torch.utils.data.random_split(test_dataset, test_split))
    # print(type(train_datasets[0]))
    for i in range(len(client_ids)):
        out_dir = f'{output_dir}/{dataset}/{client_ids[i]}'
        os.makedirs(out_dir + '/train', exist_ok=True)
        os.makedirs(out_dir + '/test', exist_ok=True)
        torch.save(train_datasets[i], out_dir + f'/train/{client_ids[i]}.pt')
        torch.save(test_datasets[i], out_dir + f'/test/{client_ids[i]}.pt')
    print('Done')

    return len(train_dataset), input_channels


def split_dataset_cifar10tl_exp(client_ids: list, datapoints, output_dir='data'):
    print('Splitting dataset (may take some time)...', end='')

    num_clients = len(client_ids)
    train_dataset, test_dataset, input_channels = datasets.load_full_dataset("cifar10_tl", output_dir, num_clients, datapoints)
    per_client_trainset_size = datapoints
    train_split = [per_client_trainset_size]*num_clients
    train_datasets = list(torch.utils.data.random_split(train_dataset, train_split))

    for i in range(len(client_ids)):
        out_dir = f'{output_dir}/cifar10_tl/{client_ids[i]}'
        os.makedirs(out_dir + '/train', exist_ok=True)
        os.makedirs(out_dir + '/test', exist_ok=True)
        torch.save(train_datasets[i], out_dir + f'/train/{client_ids[i]}.pt')
        torch.save(test_dataset, out_dir + f'/test/{client_ids[i]}.pt')
    print('Done')

    return len(train_dataset), input_channels


def split_dataset_cifar_setting2(client_ids: list, train_dataset, test_dataset, u_datapoints = 2000, c_datapoints = 150):

    print("Unique datapoints", u_datapoints)
    print("Common datapoints", c_datapoints)

    print('Splitting dataset (may take some time)...', end='')

    num_users = len(client_ids)

    #test split 
    dict_users_test, all_idxs_test = {}, [i for i in range(len(test_dataset))]
    test_ids = set(np.random.choice(all_idxs_test, 2000, replace=False))
    for i in range(num_users):
        dict_users_test[i] = test_ids
    

    #u_datapoints are the number of datapoints with the first client 
    #c_datapoints are the number of datapoints with each of the remaining clients

    #train split 
    dict_users_train = {}

    for i in range(num_users):
        dict_users_train[i]=[]
    
    df=pd.DataFrame(list(train_dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())

    per_class_client=int(c_datapoints/num_of_classes) 
    per_class_total=per_class_client*num_users

    per_class_uclient = int(u_datapoints/num_of_classes)
    per_class_total += per_class_uclient  #2000/10 = 200

    dict_classwise={}
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)[:per_class_total]

    for j in range(num_of_classes):
        temp=list(np.random.choice(dict_classwise[j], per_class_uclient, replace = False))
        dict_users_train[0].extend(temp)
        dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   

    for i in range(1, num_users):
        
        for j in range(num_of_classes):
          temp=list(np.random.choice(dict_classwise[j], per_class_client, replace = False))
          dict_users_train[i].extend(temp)
          dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   

    return dict_users_train, dict_users_test


