import pickle
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as transforms
import random

with open('data/x_train_eye_1_1', 'rb') as f:
    x_train1 = pickle.load(f)
with open('data/y_train_eye_1_1', 'rb') as f:
    y_train1 = pickle.load(f)

with open('data/x_train_eye_2', 'rb') as f:
    x_train2 = pickle.load(f)
with open('data/y_train_eye_2', 'rb') as f:
    y_train2 = pickle.load(f)

# a_list = [1,2,3,4,1,2,1,2,3,4]
def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)


# test_1=
# rndm_idxs=list(np.random.choice( list(range(0,3500)),1000, replace = False))
x_test1=[]
y_test1=[]
x_test2=[]
y_test2=[]
# for i in range(0,2):
#     print(x_train1[i])
# # for i in range(0, 1000):
#     idx = random.randint(0, len(x_train1)) 
#     x_train3.append(x_train1[idx])
#     x_train3.append(x_train2[idx])
#     y_train3.append(y_train1[idx])    
#     y_train3.append(y_train2[idx])
#     x_train1= list(set(x_train1) -set(x_train1[idx] ))
#     x_train2= list(set(x_train2) -set(x_train2[idx] ))    
#     y_train1= list(set(y_train1) -set(y_train1[idx] ))
#     y_train2= list(set(y_train2) -set(y_train2[idx] ))

# all_players = set(range(12))
# print(type(x_train1))
# print(len(x_train1), "::", len(y_train1))
temp =  x_train1[0: 1000]
x_test1.extend(temp)

temp =  y_train1[0: 1000]
y_test1.extend(temp)

x_train1=x_train1[1000:]
y_train1=y_train1[1000:]


temp =  x_train2[0: 1000]
x_test2.extend(temp)
temp =  y_train2[0: 1000]
y_test2.extend(temp)

x_train2=x_train2[1000:]
y_train2= y_train2[1000:]

print(len(x_train1))
# print(x_train1[0])
print(len(y_train1))
# print(y_train1[0])

print(len(x_train2))
# print(x_train2[0])
print(len(y_train2))
# print(y_train2[0])


list_0=find_indices(y_train1,0)
list_1=find_indices(y_train1,1)
list_2=find_indices(y_train1,2)
print(len(list_0), "::", len(list_1), "::", len(list_2))
print(len(y_train1))

list_0=find_indices(y_train2,0)
list_1=find_indices(y_train2,1)
list_2=find_indices(y_train2,2)
print(len(list_0), "::", len(list_1), "::", len(list_2))
print(len(y_train2))

list_0=find_indices(y_test1,0)
list_1=find_indices(y_test1,1)
list_2=find_indices(y_test1,2)
print(len(list_0), "::", len(list_1), "::", len(list_2))
print(len(y_test1))

list_0=find_indices(y_test2,0)
list_1=find_indices(y_test2,1)
list_2=find_indices(y_test2,2)
print(len(list_0), "::", len(list_1), "::", len(list_2))
print(len(y_test2))


class CreateDataset(Dataset):
    def __init__(self, x,y,idxs,  transform=None):
        super().__init__()
        self.x=x
        self.y=y
        self.transform = transform
        self.idxs=list(idxs)

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        image,label = self.x[self.idxs[index]], self.y[self.idxs[index]]
        
        if self.transform is not None:
            image = self.transform(image)
       
        return image, label

def get_eye_data(idxs, num):
    if(num==1):
        y_train=y_train1
        x_train=x_train1
    elif(num==2):
        y_train=y_train2
        x_train=x_train2
    elif(num==3):
        y_train=y_test1
        x_train=x_test1
    elif(num==4):
        y_train=y_test2
        x_train=x_test2
    transform_train= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.4),
        #transforms.ColorJitter(brightness=2, contrast=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_data = CreateDataset(x=x_train, y=y_train, idxs=idxs, transform=transform_train)
    return train_data 



def get_idxs():
    
    l1=[250,180,70]
    l2=[370,109,21]
    # l3=[295,157,48]
    dict_users, dict_users_test={}, {}
    client_id=0
    
    # dict_users_test=get_test_idxs()
    test_list=list(range(0,1000))
    for num in range(0,2):
        data_id=num+1
        if(num==0):
            y_train=y_train1
            x_train=x_train1
            l=l1
        elif(num==1):
            y_train=y_train2
            x_train=x_train2
            l=l2

        list_0=find_indices(y_train,0)
        list_1=find_indices(y_train,1)
        list_2=find_indices(y_train,2)
        print(len(list_0), "::", len(list_1), "::", len(list_2))
        # p
        for i in range(0,5):
            train_idxs=[]

            
            temp=list(np.random.choice(list_0,l[0] , replace = False))
            train_idxs.extend(temp)
            list_0 = list(set(list_0) -set( temp))


            temp=list(np.random.choice(list_1,l[1] , replace = False))
            train_idxs.extend(temp)
            list_1 = list(set(list_1) -set( temp))


            temp=list(np.random.choice(list_2,l[2] , replace = False))
            train_idxs.extend(temp)
            list_2 = list(set(list_2) -set( temp))


           

            dict_users[client_id]=train_idxs
            dict_users_test[client_id]=test_list
           
            print("data id: ", data_id, "client_id: ", client_id)
            print(len(list_0), "::", len(list_1), "::", len(list_2))
            client_id+=1
           


    return dict_users, dict_users_test

    



