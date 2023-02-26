import pandas as pd
import numpy as np


def setting2(train_full_dataset,test_full_dataset, num_users):
    dict_users, dict_users_test = {}, {}
    for i in range(num_users):
        dict_users[i]=[]
        dict_users_test[i]=[]
    
    df=pd.DataFrame(list(train_full_dataset), columns=['images', 'labels'])
    df_test=pd.DataFrame(list(test_full_dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())

    dict_classwise={}
    dict_classwise_test={}
    
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)

    for i in range(num_of_classes):
      dict_classwise_test[i] = df_test[df_test['labels']==i].index.values.astype(int)

    for i in range(num_users):
        
        for j in range(num_of_classes):
            if(i==j or (i+1)%10==j):
                temp=list(np.random.choice(dict_classwise[j], 225, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
            
            elif((i+2)%10==j or (i+3)%10==j):
                temp=list(np.random.choice(dict_classwise[j], 7, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))

            else:
                temp=list(np.random.choice(dict_classwise[j],6, replace = False))
                dict_users[i].extend(temp)
                dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   
    for i in range(num_users):
        
        for j in range(num_of_classes):
            if(i==j or (i+1)%10==j):
                temp=list(np.random.choice(dict_classwise_test[j],450 , replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
            elif((i+2)%10==j or (i+3)%10==j or (i+4)%10==j or (i+5)%10==j):
                temp=list(np.random.choice(dict_classwise_test[j], 13, replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
            else:
                temp=list(np.random.choice(dict_classwise_test[j], 12, replace = False))
                dict_users_test[i].extend(temp)
                dict_classwise_test[j] = list(set(dict_classwise_test[j]) -set( temp))
   
   
    return dict_users , dict_users_test           


def setting1(dataset, num_users, datapoints):
    
    dict_users = {}
    
    for i in range(num_users):
        dict_users[i]=[]
    df=pd.DataFrame(list(dataset), columns=['images', 'labels'])
    num_of_classes=len(df['labels'].unique())
    
    per_class_client=int(datapoints/num_of_classes)
    per_class_total=per_class_client*num_users

    dict_classwise={}
 
    for i in range(num_of_classes):
      dict_classwise[i] = df[df['labels']==i].index.values.astype(int)[:per_class_total]

    for i in range(num_users):
        
        for j in range(num_of_classes):
          temp=list(np.random.choice(dict_classwise[j], per_class_client, replace = False))
          dict_users[i].extend(temp)
          dict_classwise[j] = list(set(dict_classwise[j]) -set( temp))
   
    return dict_users              

def get_test_dict(dataset, num_users):
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    val = len(dataset)//num_users
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, val, replace=False))
    return dict_users            


def get_dicts(train_full_dataset, test_full_dataset,  num_users, setting, datapoints):
    
    if setting == 'setting2':
        dict_users, dict_users_test=setting2(train_full_dataset, test_full_dataset, num_users)

    elif setting == 'setting1':
        dict_users = setting1(train_full_dataset, num_users, datapoints)
        dict_users_test=get_test_dict(test_full_dataset, num_users)
        
    return dict_users, dict_users_test