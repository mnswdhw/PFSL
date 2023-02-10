import torch
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

import matplotlib.pyplot as plt
import wandb
import copy
import argparse
from utils import datasets, dataset_settings
import time
from sklearn.metrics import classification_report

matplotlib.use('Agg')

## ARGPARSER

def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Naive Split leaning",
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
        metavar="B",
        help="Batch size",
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
        help='NUmber of data samples per client in setting 1'
     
    )
    parser.add_argument(
        "--disable_wandb",
        type=bool,
        default=False,
        help='Disable wandb'
     
    )
    parser.add_argument(
        "--opt_iden",
        type=str,
        default="",
        help="optional identifier of experiment",
    )

    args = parser.parse_args()
    return args




#=====================================================================================================
#                           Client-side Model definition
#=====================================================================================================
# Model at client side
class ResNet18_client_side(nn.Module):
    def __init__(self,input_channels):
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
# Model at server side
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
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc



# Server-side function associated with Training 
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user
    
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr = lr)

    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_glob_server(fx_client)
    
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
    
    # server-side model net_glob_server is global so it is updated automatically in each pass to this function
        # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train)/len(batch_acc_train)           # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train)/len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count1 = 0
        
        if idx == 0:
            prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train, loss_avg_train))
            
        
                
        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch-1:
            
            l_epoch_check = True                # for evaluate_server function - to check local epoch has hitted 
                       
            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)
            
            #print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train
                        
            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)
            all_clients_train_acc[idx].append(acc_avg_train_all)
            
            # collect the id of each new user                        
            if idx not in idx_collect:
                idx_collect.append(idx) 
                #print(idx_collect)
        
        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_users:
            fed_check = True                                                  # for evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display 
                        
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
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train
    global targets, outputs, mycount, max_f1, max_epoch, macro_avg_f1_3classes, macro_avg_f1_dict

    net_glob_server.eval()
  
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device) 
        #---------forward prop-------------
        fx_server = net_glob_server(fx_client)
        
        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy

        _,pred_t = torch.max(fx_server, dim=1)
     
        outputs.extend(pred_t.cpu().detach().numpy().tolist())
        targets.extend(y.cpu().detach().numpy().tolist())
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
            
            # if idx == 0:
            prGreen('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test, loss_avg_test))
            # unique_test.append(acc_avg_test)
            

            
            # if a local epoch is completed   
            if l_epoch_check:
                l_epoch_check = False

                clr=classification_report(np.array(targets), np.array(outputs), output_dict=True)
                # macro_avg_f1_3classes.append((clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2)
                curr_f1=(clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2
                macro_avg_f1_3classes.append(curr_f1)
                macro_avg_f1_dict[idx]=curr_f1
                
                targets=[]
                outputs=[]
                
                # Store the last accuracy and loss
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test
                
                        
                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)
                all_clients_test_acc[idx].append(acc_avg_test_all)
             
                
            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False
                                
                acc_avg_all_user = sum(acc_test_collect_user)/len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user)/len(loss_test_collect_user)
                f1_avg_all_user=sum(macro_avg_f1_3classes)/len(macro_avg_f1_3classes)
            
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user= []
                macro_avg_f1_3classes=[]

                              
                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train, loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user, loss_avg_all_user))
                print("==========================================================")

                # print("all clients test accuracy ", all_clients_test_acc)
              
                if(f1_avg_all_user> max_f1):
                    max_f1=f1_avg_all_user
                    max_epoch=ell

                    print("MAX F1: ", max_f1, "MAX EPOCh: ", max_epoch)
                wandb.log({
                "Epoch": ell,
                "F1 Scores: ": f1_avg_all_user,
                "Personalized Average Train Accuracy": acc_avg_all_user_train,
                "Personalized Average Test Accuracy": acc_avg_all_user,  
               
            })
                macro_avg_f1_dict={}
                
    return 

#==============================================================================================================
#                                       Clients Side Program
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
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1 
        #self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = args.batch_size, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = args.test_batch_size, shuffle = True)
        # client[idx]=self
        

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

def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    
            

if __name__ == "__main__":


    unique_test = []
    #===================================================================  
    program = "SL ResNet18"
    print(f"---------{program}----------")            

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parse_arguments()

    SEED = args.seed
    num_users = args.number_of_clients
    epochs = args.epochs
    frac = args.fac      
    lr = args.lr
    dataset = args.dataset

    if dataset == "mnist" or dataset == "fmnist":
        input_channels = 1
    else:
        input_channels = 3

    no_classes=10

    

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)



    # To print in color -------test/train of the client side
    def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
    def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))   

    global outputs, targets
    targets=[]
    outputs=[]

    global clients
    clients={}

    mode="online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning_exps", mode = mode)
    wandb.run.name = args.opt_iden

    config = wandb.config          
    config.batch_size = args.batch_size    
    config.test_batch_size = args.test_batch_size        
    config.epochs = args.epochs             
    config.lr = args.lr       
    config.dataset = args.dataset
    # config.model = args.model
    config.seed = args.seed
    config.opt = args.opt_iden




    net_glob_client = ResNet18_client_side(input_channels)
    net_glob_client.to(device)
    print(net_glob_client)     


    net_glob_server = ResNet18_server_side(Baseblock, [2,2,2], no_classes) 
    net_glob_server.to(device)
    print(net_glob_server)      

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


    #client idx collector
    idx_collect = []
    l_epoch_check = False
    fed_check = False

    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(dataset, "data", num_users, args.datapoints, False)

    #----------------------------------------------------------------
    if(args.setting=='setting4'):
        dict_users = dataset_iid(train_full_dataset, num_users)
        dict_users_test = dataset_iid(test_full_dataset, num_users)
    else:
        dict_users , dict_users_test = dataset_settings.get_dicts(train_full_dataset, test_full_dataset, num_users, args.setting, args.datapoints)
  
    print("len: ", len(dict_users[0]))
    
    #net_glob_client.train()
    # this epoch is global epoch, also known as rounds
    macro_avg_f1_3classes=[]
    macro_avg_f1_dict={}
    all_clients_test_acc=dict()
    all_clients_train_acc=dict()
    for idx in range(0,num_users):
        
        all_clients_test_acc[idx]=[]
        all_clients_train_acc[idx]=[]

    st = time.time()
    max_epoch, max_f1, max_accuracy=0,0,0
    for iter in range(epochs):
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace = False)

        # Sequential training/testing among clients      
        for idx in idxs_users:
            local = Client(net_glob_client, idx, lr, device, dataset_train = train_full_dataset, dataset_test = test_full_dataset, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------
            w_client = local.train(net = copy.deepcopy(net_glob_client).to(device))
                
            # Testing -------------------
            local.evaluate(net = copy.deepcopy(net_glob_client).to(device), ell= iter)
            
            # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
            net_glob_client.load_state_dict(w_client)
    
    #===================================================================================     

    et = time.time()
    print("Training and Evaluation completed!")    
    print(f"Time taken {(et-st)/60} mins")
    # print("Max test accuracy of unique client is: ", max(unique_test))   
    #===============================================================================
    # Save output data to .excel file (we use for comparision plots)
    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
    file_name = f"results/SL/{program}_{args.batch_size}_{args.dataset}_{args.lr}_{args.epochs}_setting2"+".xlsx"    
    df.to_excel(file_name, sheet_name= f"v1_test", index = False)     

    X = range(args.epochs)
    all_clients_stacked_test  = np.array([test_acc for _,test_acc in all_clients_test_acc.items()])
    all_clients_stacked_train = np.array([train_acc for _,train_acc in all_clients_train_acc.items()])
   

    # all_clients_stacked_train = np.array([client.train_acc for _,client in clients.items()])
    # all_clients_stacked_test = np.array([client.test_acc for _,client in clients.items()])
    epochs_train_std = np.std(all_clients_stacked_train,axis = 0, dtype = np.float64)
    epochs_test_std = np.std(all_clients_stacked_test,axis = 0, dtype = np.float64)
    

    #Y_train is the average client train accuracies at each epoch
    #epoch_train_std is the standard deviation of clients train accuracies at each epoch
   
    Y_train = acc_train_collect
    Y_train_lower = Y_train - (1.65 * epochs_train_std) #95% of the values lie between 1.65*std
    Y_train_upper = Y_train + (1.65 * epochs_train_std)

    Y_test = acc_test_collect
    Y_test_lower = Y_test - (1.65 * epochs_test_std) #95% of the values lie between 1.65*std
    Y_test_upper = Y_test + (1.65 * epochs_test_std)

    Y_train_cv =  epochs_train_std / Y_train
    Y_test_cv = epochs_test_std / Y_test

    plt.figure(0)
    plt.plot(X, Y_train)
    plt.fill_between(X,Y_train_lower , Y_train_upper, color='blue', alpha=0.25)
    # plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
   
    plt.show()
    wandb.log({"train_plot": wandb.Image(plt)})

    plt.figure(1)
    plt.plot(X, Y_test)
    plt.fill_between(X,Y_test_lower , Y_test_upper, color='blue', alpha=0.25)
    # plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    plt.show()
    wandb.log({"test_plot": wandb.Image(plt)})

    plt.figure(2)
    plt.plot(X, Y_train_cv)
    plt.show()
    wandb.log({"train_cv": wandb.Image(plt)})

    plt.figure(3)
    plt.plot(X, Y_test_cv)
    plt.show()
    wandb.log({"test_cv": wandb.Image(plt)})

    #=============================================================================
    #                         Program Completed
    #============================================================================= 

