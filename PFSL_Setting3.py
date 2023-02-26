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
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp, split_dataset_cifar_setting2
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import multiprocessing
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
import pandas as pd
import time 
from utils.split_dataset import DatasetFromSubset
from utils import datasets,dataset_settings
import torch.nn.functional as F

#Helper function to load train and test data for each client
class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client_try():
    def __init__(self, id, *args, **kwargs):
        super(Client_try, self).__init__(*args, **kwargs)
        self.id = id
        self.front_model = []
        self.back_model = []
        self.losses = []
        self.train_dataset = None
        self.test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.train_iterator = None
        self.test_iterator = None 
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None
        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.train_acc = []
        self.test_acc = []
        self.front_epsilons = []
        self.front_best_alphas = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')


    def backward_back(self):
        self.loss.backward()


    def backward_front(self):
        print(self.remote_activations1.grad)
        self.activations1.backward(self.remote_activations1.grad)


    def calculate_loss(self):
        self.criterion = F.cross_entropy
        self.loss = self.criterion(self.outputs, self.targets)


    def calculate_test_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.test_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.test_acc[-1]}')


    def calculate_train_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.train_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.train_acc[-1]}')

    def create_DataLoader(self, dataset_train,dataset_test,idxs,idxs_test, batch_size, test_batch_size):
        self.train_DataLoader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs), batch_size = batch_size, shuffle = True)
        self.test_DataLoader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = test_batch_size, shuffle = True)

    def forward_back(self):
        self.back_model.to(self.device)
        self.outputs = self.back_model(self.remote_activations2)


    def forward_front(self, type, u_id = None):

        if type == "train":
            
            if self.id == u_id:

                try:
                    self.data, self.targets = next(self.train_iterator)
                except StopIteration:
                    self.train_iterator = iter(self.train_DataLoader)
                    self.data, self.targets = next(self.train_iterator)
            else:
                self.data, self.targets = next(self.train_iterator)

        else:
            self.data, self.targets = next(self.test_iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
        self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        self.remote_activations1 = self.activations1.detach().requires_grad_(True)


    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']

    def idle(self):
        pass


    def load_data(self, dataset, transform):
        try:
            dataset_path = os.path.join(f'data/{dataset}/{self.id}')
        except:
            raise Exception(f'Dataset not found for client {self.id}')
        self.train_dataset = torch.load(f'{dataset_path}/train/{self.id}.pt')
        self.test_dataset = torch.load('data/cifar10_setting2/test/common_test.pt')

        self.train_dataset = DatasetFromSubset(
            self.train_dataset, transform=transform
        )
        self.test_dataset = DatasetFromSubset(
            self.test_dataset, transform=transform
        )


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()


    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()




def generate_random_client_ids_try(num_clients, id_len=4) -> list:
    client_ids = []
    for _ in range(num_clients):
        client_ids.append(''.join(random.sample("abcdefghijklmnopqrstuvwxyz1234567890", id_len)))
    return client_ids 

def generate_random_clients_try(num_clients) -> dict:
    client_ids = generate_random_client_ids_try(num_clients)
    clients = {}
    for id in client_ids:
        clients[id] = Client_try(id)
    return clients



def initialize_client(client, dataset_train,dataset_test,idxs,idxs_test, batch_size, test_batch_size):
     client.create_DataLoader(dataset_train,dataset_test, idxs, idxs_test, batch_size, test_batch_size)

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
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  

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

    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution

if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)


    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients_try(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(args.dataset, "data", args.number_of_clients, args.datapoints, args.pretrained)
    dict_users_train , dict_users_test = split_dataset_cifar_setting2(client_ids, train_full_dataset, test_full_dataset)

    transform=None

    #Assigning train and test data to each client depending for each client
    print('Initializing clients...')
    for i,(_, client) in enumerate(clients.items()):
        initialize_client(client, train_full_dataset, test_full_dataset, dict_users_train[i], dict_users_test[i], args.batch_size, args.test_batch_size)
   
    print('Client Intialization complete.')
    # Train and test data intialisation complete

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

    common_client = clients[client_ids[1]]
    unique_client_id = client_ids[0]
    unique_client = clients[unique_client_id]
    num_iterations_common = ceil(len(common_client.train_DataLoader.dataset)/args.batch_size)
    num_test_iterations = ceil(len(common_client.test_DataLoader.dataset)/args.test_batch_size)
    unique_client_iterations = ceil(len(unique_client.train_DataLoader.dataset)/args.batch_size)
    unique_client.train_iterator = iter(unique_client.train_DataLoader)

    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_model = model.center(pretrained=args.pretrained)
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)


    st = time.time()

    max_train_small_clients = 0
    max_test_small_clients = 0
    max_train_large_client = 0
    max_test_large_client = 0

    #Starting the training process 
    for epoch in range(args.epochs):

        overall_train_acc.append(0)
        for i,(_, client) in enumerate(clients.items()):
            client.train_acc.append(0)
            if i != 0:
                client.train_iterator = iter(client.train_DataLoader)
            
        for c_id, client in clients.items():
            #For every batch in the current epoch
            for iteration in range(num_iterations_common):

                
                if c_id == unique_client_id:
                    client.forward_front("train", u_id = c_id)
                else:
                    client.forward_front("train")

                sc_clients[c_id].remote_activations1 = clients[c_id].remote_activations1
                sc_clients[c_id].forward_center()
                client.remote_activations2 = sc_clients[c_id].remote_activations2
                client.forward_back()
                client.calculate_loss()
                client.backward_back()
                sc_clients[c_id].remote_activations2 = clients[c_id].remote_activations2
                sc_clients[c_id].backward_center()
                    
                # client.remote_activations1 = copy.deepcopy(sc_clients[client_id].remote_activations1)
                # client.backward_front()

                client.step_back()
                client.zero_grad_back()
                sc_clients[c_id].center_optimizer.step()
                sc_clients[c_id].center_optimizer.zero_grad()
                client.train_acc[-1] += client.calculate_train_acc()

            client.train_acc[-1] /= num_iterations_common
            if c_id == unique_client_id:
                print("Large client train accuracy", client.train_acc[-1])
                if client.train_acc[-1] >= max_train_large_client:
                    max_train_large_client = client.train_acc[-1]
            else:    
                overall_train_acc[-1] += client.train_acc[-1]

        overall_train_acc[-1] /= (len(clients)-1)
        if overall_train_acc[-1] >= max_train_small_clients:
            max_train_small_clients = overall_train_acc[-1]
        print(f'Epoch {epoch} C1-C10 Average Train Acc: {overall_train_acc[-1]}\n')

        # merge weights below uncomment 
        params = []
        for _, client in sc_clients.items():
            params.append(copy.deepcopy(client.center_model.state_dict()))
        w_glob = merge_weights(params)

        for _, client in sc_clients.items():
            client.center_model.load_state_dict(w_glob)

        params = []
        for _, client in clients.items():
            params.append(copy.deepcopy(client.back_model.state_dict()))
        w_glob_cb = merge_weights(params)

        for _, client in clients.items():
            client.back_model.load_state_dict(w_glob_cb)


        


        # Testing on every 5th epoch
        if epoch%5 == 0:
            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)
                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.test_iterator = iter(client.test_DataLoader)

                for client_id, client in clients.items():
                    for iteration in range(num_test_iterations):
        
                        client.forward_front("test")  
                        sc_clients[client_id].remote_activations1 = clients[client_id].remote_activations1
                        sc_clients[client_id].forward_center()
                        client.remote_activations2 = sc_clients[client_id].remote_activations2
                        client.forward_back()
                        client.test_acc[-1] += client.calculate_test_acc()

                    client.test_acc[-1] /= num_test_iterations
                    if client_id == unique_client_id:
                        print("Large client test accuracy", client.test_acc[-1])
                        if client.test_acc[-1] >= max_test_large_client:
                            max_test_large_client = client.train_acc[-1]
                    else:
                        overall_test_acc[-1] += client.test_acc[-1]  #not including test accuracy of unique client

                overall_test_acc[-1] /= (len(clients)-1)

                if overall_test_acc[-1] >= max_test_small_clients:
                    max_test_small_clients = overall_test_acc[-1]
                print(f' Epoch {epoch} C1-C10 Average Test Acc: {overall_test_acc[-1]}\n')

    # print("Average C1 - C10 Train accuracy: ", max_train_small_clients)
    # print("Large Client train Accuracy: ", max_train_large_client)
    print("Average C1 - C10 test accuracy: ", max_test_small_clients)
    print("Large Client test Accuracy: ", max_test_large_client)
            
        


    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    '''

    et = time.time()
    print(f"Time taken for this run {(et - st)/60} mins")

  
    #BELOW CODE TO PLOT MULTIPLE LINES ON A SINGLE PLOT ONE LINE FOR EACH CLIENT
    # for client_id, client in clients.items():
    #     plt.plot(list(range(args.epochs)), client.train_acc, label=f'{client_id} (Max:{max(client.train_acc):.4f})')
    # plt.plot(list(range(args.epochs)), overall_train_acc, label=f'Average (Max:{max(overall_train_acc):.4f})')
    # plt.title(f'{args.number_of_clients} Clients: Train Accuracy vs. Epochs')
    # plt.ylabel('Train Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.ioff()
    # plt.savefig(f'./results/train_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    # plt.show()

    # for client_id, client in clients.items():
    #     plt.plot(list(range(args.epochs)), client.test_acc, label=f'{client_id} (Max:{max(client.test_acc):.4f})')
    # plt.plot(list(range(args.epochs)), overall_test_acc, label=f'Average (Max:{max(overall_test_acc):.4f})')
    # plt.title(f'{args.number_of_clients} Clients: Test Accuracy vs. Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.ioff()
    # plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    # plt.show()
