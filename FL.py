#===========================================================
# Federated learning: ResNet18 
# ===========================================================
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob 
import math
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import argparse
from utils import datasets,dataset_settings
import time
import wandb
from sklearn.metrics import classification_report


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
        "--setting",
        type=str,
        default="setting1",
        help='Setting you would like to run for, i.e, setting1 , setting2 or setting4'
     
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=512,
        help="Input batch size for testing"

    )


    parser.add_argument(
        "--datapoints" ,
        type=int,
        default=500,
        help='Number of data samples per client in setting 1',
    )
    args = parser.parse_args()
    args = parser.parse_args()
    return args

#==============================================================================================================
#                                  Client Side Program 
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
class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = args.batch_size, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = args.test_batch_size, shuffle = True)
        clients[idx]=self

    def train(self, net):
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr = self.lr, momentum = 0.5)
        optimizer = torch.optim.Adam(net.parameters(), lr = self.lr)
        
        epoch_acc = []
        epoch_loss = []
        for iter in range(self.local_ep):
            batch_acc = []
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                
                #--------backward prop--------------
                loss.backward()
                optimizer.step()
                              
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            # if self.idx == 0:
            prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,
                            iter, acc.item(), loss.item()))
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
    

    def evaluate(self, net, ell):
        global targets, outputs
        net.eval()
           
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                _,pred_t = torch.max(fx, dim=1)
                outputs.extend(pred_t.cpu().detach().numpy().tolist())
                targets.extend(labels.cpu().detach().numpy().tolist())
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                                 
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            prGreen('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))

            clr=classification_report(np.array(targets), np.array(outputs), output_dict=True)
           
            curr_f1=(clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2
            macro_avg_f1_3classes.append(curr_f1)
            macro_avg_f1_dict[idx]=curr_f1
            targets=[]
            outputs=[]

            # if self.idx == 0:
            unique_test.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)

#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client

def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users    

#====================================================================================================
#                               Server Side Program
#====================================================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 
# building a ResNet18 Architecture
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18(nn.Module):

    def __init__(self, block, layers, input_channels, num_classes=1000):
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


#===========================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
#====================================================


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
    plt.show()
    # wandb.log({"Histogram": wandb.Image(plt)})
    plt.savefig('plot_fl.png')
    # plt.savefig(f'./results/classvsfreq/settin3{dataset}.png')  

    max_len=0
    #plot line graphs
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
        max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    plt.xticks(np.arange(0,10))
    plt.ylim(0, max_len)
    plt.legend()
    plt.show()
    # wandb.log({"Line graph": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_fre/q/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution



if __name__ == "__main__":


    unique_test = []
    #===================================================================  
    program = "FL ResNet18"
    print(f"---------{program}----------")            

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_arguments()

    SEED = args.seed
    num_users = args.number_of_clients
    epochs = args.epochs
    frac = args.fac      
    lr = args.lr
    dataset = args.dataset

    if args.dataset == "mnist" or args.dataset == "fmnist":
        input_channels = 1
    else:
        input_channels = 3
    no_classes=10

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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

    # To print in color during test/train 
    def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
    def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    

   

    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(dataset, "data", num_users, args.datapoints, False)

    #----------------------------------------------------------------
    if(args.setting=='setting4'):
        dict_users = dataset_iid(train_full_dataset, num_users)
        dict_users_test = dataset_iid(test_full_dataset, num_users)
    else:
        dict_users , dict_users_test = dataset_settings.get_dicts(train_full_dataset, test_full_dataset, num_users, args.setting, args.datapoints)
  

    net_glob = ResNet18(BasicBlock, [2, 2, 2, 2],input_channels, no_classes) 
    net_glob.to(device)
    print(net_glob)   
    
    
    net_glob.train()
    

    w_glob = net_glob.state_dict()

    loss_train_collect = []
    acc_train_collect = []
    loss_test_collect = []
    acc_test_collect = []
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
        w_locals, loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test, macro_avg_f1_3classes  =[], [], [], [], [], []
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace = False)
        
        # Training/Testing simulation
        for idx in idxs_users: # each client

            local = LocalUpdate(idx, lr, device, dataset_train = train_full_dataset, dataset_test = test_full_dataset, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
            # Training ------------------
            w, loss_train, acc_train = local.train(net = copy.deepcopy(net_glob).to(device))
            w_locals.append(copy.deepcopy(w))
            loss_locals_train.append(copy.deepcopy(loss_train))
            acc_locals_train.append(copy.deepcopy(acc_train))
            all_clients_train_acc[idx].append(acc_train)
            # Testing -------------------
            loss_test, acc_test = local.evaluate(net = copy.deepcopy(net_glob).to(device), ell=iter)
            loss_locals_test.append(copy.deepcopy(loss_test))
            acc_locals_test.append(copy.deepcopy(acc_test))
            all_clients_test_acc[idx].append(acc_test)

        # if iter==0:
        #     client_ids=[0,1,2,3,4,5,6,7,8,9]
        #     plot_class_distribution(clients, client_ids)  
            
        # Federation process
        w_glob = FedAvg(w_locals)
        print("------------------------------------------------")
        print("------ Federation process at Server-Side -------")
        print("------------------------------------------------")
        
        # update global model --- copy weight to net_glob -- distributed the model to all users
        net_glob.load_state_dict(w_glob)
        
        # Train/Test accuracy
        acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
        acc_train_collect.append(acc_avg_train)
        acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
        acc_test_collect.append(acc_avg_test)
        

        f1_avg_all_user=sum(macro_avg_f1_3classes)/ len(macro_avg_f1_3classes)
        if(args.setting=='setting2'):
            if(f1_avg_all_user> max_f1):
                max_f1=f1_avg_all_user
                max_epoch=iter
                print("Max F1Score: ", max_f1)
        else:
            if(acc_avg_test> max_accuracy):
                max_accuracy=acc_avg_test
                max_epoch=iter
                print("Max Accuracy: ", max_accuracy)
        
        # Train/Test loss
        loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
        loss_train_collect.append(loss_avg_train)
        loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
        loss_test_collect.append(loss_avg_test)
        
        
        
        print('------------------- SERVER ----------------------------------------------')
        print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
        print('Test:  Round {:3d}, Avg Accuracy {:.3f} |  Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))
        if(args.setting=='setting2'):
            print('Avg F1 Score{:.3f}'.format( f1_avg_all_user ))
        print('-------------------------------------------------------------------------')

        
        wandb.log({
                "Epoch": iter,
                "F1 Scores:": f1_avg_all_user,
                
                "Personalized Average Train Accuracy": acc_avg_train,
                "Personalized Average Test Accuracy": acc_avg_test,  
           
            })
        macro_avg_f1_dict={}
    

    #===================================================================================     

    print("Training and Evaluation completed!")    
    et = time.time()
    print(f"Total time taken is {(et-st)/60} mins")
    print("Max test accuracy of unique client is: ", max(unique_test))
    #===============================================================================
    # Save output data to .excel file (we use for comparision plots)



    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})     
    file_name = f"results/FL/{program}_{args.batch_size}_{args.dataset}_{args.lr}_{args.epochs}_{args.setting}"+".xlsx"  
    df.to_excel(file_name, sheet_name= "v1_test", index = False)   

   
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