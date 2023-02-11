import os
import random
import string
import socket
import requests
import sys
import threading
import time
import torch
from math import ceil
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import multiprocessing
from sklearn.metrics import classification_report
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
import wandb
import pandas as pd
import time 
from utils import dataset_settings, datasets
import torch.nn.functional as F


#To load train and test data for each client for setting 1 and setting 2
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


#To intialize every client with their train and test data for setting 4
def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


#Plots class distribution of train data available to each client
def plot_class_distribution(clients, dataset, batch_size, epochs, opt, client_ids):
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
    wandb.log({"Histogram": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  
    plt.savefig('plot_setting3_exp.png')

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
    wandb.log({"Line graph": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution



if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)

    #setup for wandb

    mode = "online"
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
    config.model = args.model
    config.seed = args.seed
    config.opt = args.opt_iden                              


    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    train_dataset_size, input_channels = split_dataset(args.dataset, client_ids, pretrained=args.pretrained)

    print(f'Random client ids:{str(client_ids)}')
    transform=None
    max_epoch=0
    max_f1=0

    #Assigning train and test data to each client depending for each client
    print('Initializing clients...')
    
    if(args.setting=="setting4"):
        for _, client in clients.items():
            (initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
    else:
        
        train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(args.dataset, "data", args.number_of_clients, args.datapoints, args.pretrained)
        #----------------------------------------------------------------
        dict_users , dict_users2 = dataset_settings.get_dicts(train_full_dataset, test_full_dataset, args.number_of_clients, args.setting, args.datapoints)

        dict_users_test_equal=dataset_settings.get_test_dict(test_full_dataset, args.number_of_clients)

        client_idx=0
        dict_user_train=dict()
        dict_user_test=dict()
        client_idxs=dict()

        for _, client in clients.items():
            dict_user_train[_]=dict_users[client_idx]
            dict_user_test[_]=dict_users2[client_idx]
            client_idxs[_]=client_idx
            client_idx+=1
        for _, client in clients.items():
            client.train_dataset=DatasetSplit(train_full_dataset, dict_user_train[_])
            client.test_dataset=DatasetSplit(test_full_dataset, dict_user_test[_])
            client.create_DataLoader(args.batch_size, args.test_batch_size)
    print('Client Intialization complete.')    
    # Train and test data intialisation complete

    #Setting the start of personalisation phase
    if(args.setting!='setting2'):
        args.checkpoint=args.epochs+10

    
    # class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)


    #Assigning front, center and back models and their optimizers for all the clients
    model = importlib.import_module(f'models.{args.model}')
  
    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
    print('Done')
  
    for _, client in clients.items():
        # client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)
        # client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=args.lr, momentum=0.9)
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)

    first_client = clients[client_ids[0]]
    num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)
   
    num_test_iterations= ceil(len(first_client.test_DataLoader.dataset)/args.test_batch_size)
    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_model = model.center(pretrained=args.pretrained)
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)

    st = time.time()

    macro_avg_f1_2classes=[]

    criterion=F.cross_entropy

    #logging the gradients of the models of all the three parts to wandb
    for _, client in clients.items(): 
        wandb.watch(client.front_model, criterion, log="all",log_freq=2) 
        wandb.watch(client.back_model, criterion, log="all", log_freq=2)
    for _, s_client in sc_clients.items():
        wandb.watch(s_client.center_model, criterion, log="all", log_freq=2)

    #Starting the training process 
    for epoch in range(args.epochs):
        if(epoch==args.checkpoint): # When starting epoch of the perosnalisation is reached, freeze all the layers of the center model 
            for _, s_client in sc_clients.items():
                s_client.center_model.freeze(epoch, pretrained=True)

        overall_train_acc.append(0)


        for _, client in clients.items():
            client.train_acc.append(0)
            client.iterator = iter(client.train_DataLoader)
            
        #For every batch in the current epoch
        for iteration in range(num_iterations):
            print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')

            
            for _, client in clients.items():
                client.forward_front()

            for client_id, client in sc_clients.items():
                client.remote_activations1 = clients[client_id].remote_activations1
                client.forward_center()

            for client_id, client in clients.items():
                client.remote_activations2 = sc_clients[client_id].remote_activations2
                client.forward_back()

            for _, client in clients.items():
                client.calculate_loss()

            for _, client in clients.items():
                client.backward_back()

            for client_id, client in sc_clients.items():
                client.remote_activations2 = clients[client_id].remote_activations2
                client.backward_center()

            for _, client in clients.items():
                client.step_back()
                client.zero_grad_back()

            #merge grads uncomment below

            # if epoch%2 == 0:
            #     params = []
            #     normalized_data_sizes = []
            #     for iden, client in clients.items():
            #         params.append(sc_clients[iden].center_model.parameters())
            #         normalized_data_sizes.append(len(client.train_dataset) / train_dataset_size)
            #     merge_grads(normalized_data_sizes, params)

            for _, client in sc_clients.items():
                client.center_optimizer.step()
                client.center_optimizer.zero_grad()

            for _, client in clients.items():
                client.train_acc[-1] += client.calculate_train_acc() #train accuracy of every client in the current epoch in the current batch

        for c_id, client in clients.items():
            client.train_acc[-1] /= num_iterations # train accuracy of every client of all the batches in the current epoch
            overall_train_acc[-1] += client.train_acc[-1] 

        overall_train_acc[-1] /= len(clients) #avg train accuracy of all the clients in the current epoch
        print(f' Personalized Average Train Acc: {overall_train_acc[-1]}')

        # merge weights below uncomment 
        params = []
        for _, client in sc_clients.items():
            params.append(copy.deepcopy(client.center_model.state_dict()))
        w_glob = merge_weights(params)

        for _, client in sc_clients.items():
            client.center_model.load_state_dict(w_glob)

        params = []

        #In the personalisation phase merging of weights of the back layers is stopped
        if(epoch <=args.checkpoint):
            for _, client in clients.items():
                params.append(copy.deepcopy(client.back_model.state_dict()))
            w_glob_cb = merge_weights(params)
            del params
    
            for _, client in clients.items():
                client.back_model.load_state_dict(w_glob_cb)

        #Testing every epoch
        if (epoch%1 == 0 ):
            if(epoch==args.checkpoint):
                for _, s_client in sc_clients.items():
                    s_client.center_model.freeze(epoch, pretrained=True)
            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)
            
                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.iterator = iter(client.test_DataLoader)
                    client.pred=[]
                    client.y=[]

                #For every batch in the testing phase
                for iteration in range(num_test_iterations):
    
                    for _, client in clients.items():
                        client.forward_front()

                    for client_id, client in sc_clients.items():
                        client.remote_activations1 = clients[client_id].remote_activations1
                        client.forward_center()

                    for client_id, client in clients.items():
                        client.remote_activations2 = sc_clients[client_id].remote_activations2
                        client.forward_back()

                    for _, client in clients.items():
                        client.test_acc[-1] += client.calculate_test_acc()


                for _, client in clients.items():
                    client.test_acc[-1] /= num_test_iterations
                    overall_test_acc[-1] += client.test_acc[-1]
                    #Calculating the F1 scores using the classification report from sklearn metrics
                    if(args.setting=='setting2'):
                        clr=classification_report(np.array(client.y), np.array(client.pred), output_dict=True)
                        idx=client_idxs[_]

                        macro_avg_f1_2classes.append((clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2) #macro f1 score of the 2 prominent classes in setting2
                        
                
                overall_test_acc[-1] /= len(clients) #average test accuracy of all the clients in the current epoch

                if(args.setting=='setting2'):
                    f1_avg_all_user=sum(macro_avg_f1_2classes)/len(macro_avg_f1_2classes) #average f1 scores of the clients for the prominent 2 classes in the current epoch
                    macro_avg_f1_2classes=[]
                    print(f' Personalized Average Test Acc: {overall_test_acc[-1]}  f1 score: {f1_avg_all_user} ')

                    #Noting the maximum f1 score
                    if(f1_avg_all_user> max_f1):
                        max_f1=f1_avg_all_user
                        max_epoch=epoch
                else:
                    print(f' Personalized Average Test Acc: {overall_test_acc[-1]}   ')
            
        
            wandb.log({
                "Epoch": epoch,
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average Test Accuracy": overall_test_acc[-1],  
            })

    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    '''

    et = time.time()
    print(f"Time taken for this run {(et - st)/60} mins")
    wandb.log({"time taken by program in mins": (et - st)/60})


    # calculating the train and test standarad deviation and teh confidence intervals 
    X = range(args.epochs)
    all_clients_stacked_train = np.array([client.train_acc for _,client in clients.items()])
    all_clients_stacked_test = np.array([client.test_acc for _,client in clients.items()])
    epochs_train_std = np.std(all_clients_stacked_train,axis = 0, dtype = np.float64)
    epochs_test_std = np.std(all_clients_stacked_test,axis = 0, dtype = np.float64)

    #Y_train is the average client train accuracies at each epoch
    #epoch_train_std is the standard deviation of clients train accuracies at each epoch
    Y_train = overall_train_acc
    Y_train_lower = Y_train - (1.65 * epochs_train_std) #95% of the values lie between 1.65*std
    Y_train_upper = Y_train + (1.65 * epochs_train_std)

    Y_test = overall_test_acc
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


