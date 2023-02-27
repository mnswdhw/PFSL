import torch
from math import ceil
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
# from utils import datasets
import argparse
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from utils import get_eye_dataset


## ARGPARSER

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Splitfed V1 configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "-c",
        "--number_of_clients",
        type=int,
        default=5,
        metavar="C",
        help="Number of Clients",
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Total number of epochs to train",
    )

    parser.add_argument(
        "--fac",
        type=float,
        default= 1.0,
        metavar="N",
        help="fraction of active/participating clients, if 1 then all clients participate in SFLV1",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="States dataset to be used",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=512,

    )
    
    parser.add_argument(
        "--setting",
        type=str,
        default="setting1",
    )

    parser.add_argument(
        "--datapoints",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--opt_iden",
        type=str,
        # default=False
    )

    args = parser.parse_args()
    return args

def plot_class_distribution(clients,  client_ids):
    class_distribution=dict()
    number_of_clients=len(client_ids)
    if(len(clients)<=20):
        plot_for_clients=client_ids
    else:
        plot_for_clients=random.sample(client_ids, 20)
    
    fig, ax = plt.subplots(nrows=(int(ceil(len(client_ids)/5))), ncols=5, figsize=(15, 10))
    j=0
    i=0

    #plot histogram
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        class_distribution[client_id]=df['labels'].value_counts().sort_index()
        df['labels'].value_counts().sort_index().plot(ax = ax[i,j], kind = 'bar', ylabel = 'frequency', xlabel=client_id)
        j+=1
        if(j==5 or j==10 or j==15):
            i+=1
            j=0
    fig.tight_layout()
    # plt.show()
    plt.savefig('plot_sflv1.png')
    # plt.savefig(f'./results/classvsfreq/settin3{dataset}.png')  

    max_len=0
    #plot line graphs
    # for client_id in plot_for_clients:
    #     df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
    #     df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
    #     max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    # plt.xticks(np.arange(0,10))
    # plt.ylim(0, max_len)
    # plt.legend()
    # plt.show()
    # plt.savefig(f'./results/class_vs_fre/q/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution

#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side

import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

class ResNet18_client_side(nn.Module):
    def __init__(self, input_channels):
        super(ResNet18_client_side, self).__init__()
        self.layer1 = nn.Sequential (
                nn.Conv2d(input_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding =1),
            )
        self.layer2 = nn.Sequential  (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),              
            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def forward(self, x):
        resudial1 = F.relu(self.layer1(x))
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1 # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2
 
 
           
#=====================================================================================================
#                           Server-side Model definition
#=====================================================================================================

class Baseblock(nn.Module):
    expansion = 1
    def __init__(self, input_planes, planes, stride = 1, dim_change = None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride =  stride, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride = 1, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change
        
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        
        if self.dim_change is not None:
            res =self.dim_change(res)
            
        output += res
        output = F.relu(output)
        
        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        self.layer3 = nn.Sequential (
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU (inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
                nn.BatchNorm2d(64),       
                )   
        
        self.layer4 = self._layer(block, 128, num_layers[0], stride = 2)
        self.layer5 = self._layer(block, 256, num_layers[1], stride = 2)
        self.layer6 = self._layer(block, 512, num_layers[2], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * block.expansion, classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _layer(self, block, planes, num_layers, stride = 2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(nn.Conv2d(self.input_planes, planes*block.expansion, kernel_size = 1, stride = stride),
                                       nn.BatchNorm2d(planes*block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride = stride, dim_change = dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion
            
        return nn.Sequential(*netLayers)
        
    
    def forward(self, x):
        out2 = self.layer3(x)
        out2 = out2 + x          # adding the resudial inputs -- downsampling not required in this layer
        x3 = F.relu(out2)
        
        x4 = self. layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        
        x7 = self.avgpool(x6)
        x8 = x7.view(x7.size(0), -1) 
        y_hat =self.fc(x8)
        
        return y_hat


#====================================================================================================
#                                  Server Side Program
#====================================================================================================

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc


# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server
    global loss_train_collect_user, acc_train_collect_user, lr
    
    net_server = copy.deepcopy(net_model_server[idx]).to(device)
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)
    
    #--------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the server-side model for the current batch
    net_model_server[idx] = copy.deepcopy(net_server)
    
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        
        # copy the last trained model in the batch       
        w_server = net_server.state_dict()      
        
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # to evaluate_server function - to check local epoch has completed or not 
            # We store the state of the net_glob_server() 
            w_locals_server.append(copy.deepcopy(w_server))
            
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
        
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
        
        # This is for federation process--------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # to evaluate_server function  - to check fed check has hitted
            # Federation process at Server-Side------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                                   
            w_glob_server = FedAvg(w_locals_server)   
            
            # server-side global model update and distribute that model to all clients ------------------------------
            net_glob_server.load_state_dict(w_glob_server)    
            net_model_server = [net_glob_server for i in range(num_users)]
            
            w_locals_server = []
            idx_collect = []
            
            acc_avg_all_user_train = sum(acc_train_collect_user)/len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user)/len(loss_train_collect_user)
            
            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)
            
            acc_train_collect_user = []
            loss_train_collect_user = []
            
    # send gradients to the client               
    return dfx_client

# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, acc_avg_train_all, loss_avg_train_all, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user,acc_test_collect_user1, acc_test_collect_user2, acc_avg_all_user_train, loss_avg_all_user_train
    global targets, outputs, mycount, max_f1, max_epoch, macro_avg_f1_3classes, macro_avg_f1_dict, max_accuracy, max_c0_4_test, max_c0_f1, max_c5_9_test, max_c5_f1, max_train_accuracy
    net = copy.deepcopy(net_model_server[idx]).to(device)
    net.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
 
        _,pred_t = torch.max(fx_server, dim=1)
        
        outputs.extend(pred_t.cpu().detach().numpy().tolist())
        targets.extend(y.cpu().detach().numpy().tolist())
      
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)
        
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test)/len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test)/len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count2 = 0
            
            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False
             
                clr=classification_report(np.array(targets), np.array(outputs), output_dict=True, zero_division=0)
                curr_f1=(clr['0']['f1-score']+clr['1']['f1-score']+clr['2']['f1-score'])/3
                macro_avg_f1_3classes.append(curr_f1)
                macro_avg_f1_dict[idx]=curr_f1
                
                targets=[]
                outputs=[]
               
                
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                if(idx>=0 and idx<5):
                    acc_test_collect_user1.append(acc_avg_test_all)
                elif(idx>=5 and idx<10):
                    acc_test_collect_user2.append(acc_avg_test_all)
                
            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False
                                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                acc_avg_all_user1 = sum(acc_test_collect_user1)/len(acc_test_collect_user1)
             
                acc_avg_all_user2 = sum(acc_test_collect_user2)/len(acc_test_collect_user2)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
                f1_avg_all_user=sum(macro_avg_f1_3classes)/len(macro_avg_f1_3classes)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                # acc_test_collect1.append(acc_avg_all_user2)
                # acc_test_collect2.append(acc_avg_all_user2)


                acc_test_collect_user = []
                acc_test_collect_user1= []
                acc_test_collect_user2 = []
                loss_test_collect_user= []
                macro_avg_f1_3classes=[]

                print(f'\rEpoch: {ell}', end='')

                if(acc_avg_all_user>  max_accuracy):
                    max_accuracy=acc_avg_all_user
                    max_train_accuracy=acc_avg_all_user_train
                    max_epoch=ell
                    max_c0_f1=macro_avg_f1_dict[0]
                    max_c5_f1=macro_avg_f1_dict[5]
                    max_c0_4_test=acc_avg_all_user1
                    max_c5_9_test=acc_avg_all_user2

                macro_avg_f1_dict={}

         
    return 

#==============================================================================================================
#                                       Clients-side Program
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, client_idx,dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        #self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = args.batch_size, shuffle = True)
        # self.ldr_test = DataLoader(dataset_test, batch_size = args.test_batch_size, shuffle = True)

        if(client_idx>=0 and client_idx<=4 ):
            train_data_id=1
            test_data_id=3
        elif(client_idx>=5 and client_idx<=9):
            train_data_id=2
            test_data_id=4
       
        self.train_dataset=get_eye_dataset.get_eye_data(idxs, train_data_id)
        self.test_dataset=get_eye_dataset.get_eye_data(idxs_test, test_data_id)
        self.ldr_train = DataLoader(self.train_dataset, batch_size = args.batch_size, shuffle = True)
        self.ldr_test = DataLoader(self.test_dataset, batch_size = args.test_batch_size, shuffle = True)
        clients[idx]=self

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr = self.lr) 
        
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                
                # Sending activations to server and receiving gradients from server
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                
                #--------backward prop -------------
                fx.backward(dfx)
                optimizer_client.step()
  
            
            #prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))
           
        return net.state_dict() 
    
    def evaluate(self, net, ell):
        net.eval()
           
        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # Sending activations to server 
                evaluate_server(fx, labels, self.idx, len_batch, ell)
            
            #prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))
            
        return          
#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this
def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
                          

if __name__ == "__main__":

    #===================================================================
    program = "SFLV1 ResNet18"
    print(f"---------{program}----------")             

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()

    SEED = args.seed
    num_users = 10
    epochs = args.epochs
    frac = args.fac      
    lr = args.lr
    dataset = args.dataset

    input_channels=3
    no_classes=3

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


    net_glob_client = ResNet18_client_side(input_channels)
    net_glob_client.to(device)

    net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], no_classes) 
    net_glob_server.to(device)    

    #===================================================================================
    # For Server Side Loss and Accuracy 
    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []
    batch_acc_train = []
    batch_loss_train = []
    batch_acc_test = []
    batch_loss_test = []


    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    # to print train - test together in each round-- these are made global
    acc_avg_all_user_train = 0
    loss_avg_all_user_train = 0
    loss_train_collect_user = []
    acc_train_collect_user = []
    loss_test_collect_user = []
    acc_test_collect_user = []
    acc_test_collect_user1 = []
    acc_test_collect_user2= []
    macro_avg_f1_3classes=[]
    macro_avg_f1_dict={}
    max_f1=0
    max_epoch=0
    max_accuracy=0
    max_train_accuracy=0
    max_c0_4_test=0
    max_c0_f1=0
    max_c5_9_test=0
    max_c5_f1=0
    targets=[]
    outputs=[]
    mycount=0
    global clients
    clients={}
    

    w_glob_server = net_glob_server.state_dict()
    w_locals_server = []

    #client idx collector
    idx_collect = []
    l_epoch_check = False
    fed_check = False

    # Initialization of net_model_server and net_server (server-side model)
    net_model_server = [net_glob_server for i in range(num_users)]
    net_server = copy.deepcopy(net_model_server[0]).to(device)
    #optimizer_server = torch.optim.Adam(net_server.parameters(), lr = lr)


    #=============================================================================
    #                         Data loading 
    #============================================================================= 
 

   
    #----------------------------------------------------------------
    # dict_users , dict_users_test = dataset_settings.get_dicts(train_full_dataset, test_full_dataset, num_users, args.setting, args.datapoints)
    d1, d2=get_eye_dataset.get_idxs()
    dict_users, dict_users_test=d1, d2
  


    #------------ Training And Testing  -----------------
    net_glob_client.train()
    #copy weights
    w_glob_client = net_glob_client.state_dict()
    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds
    st = time.time()
    for iter in range(epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace = False)
        w_locals_client = []
        
        for idx in idxs_users:
            local = Client(net_glob_client, idx, lr, device,client_idx=idx, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------
            w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
            w_locals_client.append(copy.deepcopy(w_client))
            
            # Testing -------------------
            local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
      
        if(iter==0):
            client_ids=[0,1,2,3,4,5,6,7,8,9]
            plot_class_distribution(clients, client_ids)      
        # Ater serving all clients for its local epochs------------
        # Fed  Server: Federation process at Client-Side-----------
      
        w_glob_client = FedAvg(w_locals_client)   
        
        # Update client-side global model 
        net_glob_client.load_state_dict(w_glob_client)    
        
    #===================================================================================     
    et = time.time()
    print("Training and Evaluation completed!")  
    print(f"\nTime taken for this run {(et - st)/60} mins")
    print(f'Maximum Personalized Average Test Acc: {max_accuracy}  ')
    print(f'Maximum Personalized Average Train Acc: {max_train_accuracy}  ')
    print(f'Client0 F1 Scores: {max_c0_f1}')
    print(f'Client5 F1 Scores:{max_c5_f1}')
    print(f'Personalized Average Test Accuracy for Clients 0 to 4 ": {max_c0_4_test}')
    print(f'Personalized Average Test Accuracy for Clients 5 to 9": {max_c5_9_test}')      

    #===============================================================================
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
    file_name = f"results/SFLv1/{program}_{args.batch_size}_{args.dataset}_{args.lr}_{args.epochs}_d{args.datapoints}"+".xlsx"  
    df.to_excel(file_name, sheet_name= "v1_test", index = False)   





#=============================================================================
#                         Program Completed
#=============================================================================