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
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import server
import multiprocessing
# from opacus import PrivacyEngine
# from opacus.accountants import RDPAccountant
# from opacus import GradSampleModule
# from opacus.optimizers import DPOptimizer
# from opacus.validators import ModuleValidator
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


client_cluster_size = 10 
clients_drop_frac = 0.1


def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    client.load_data(args.dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


def select_random_clients(clients):
    random_clients = {}
    client_ids = list(clients.keys())
    random_index = random.randint(0,len(client_ids)-1)
    random_client_ids = client_ids[random_index]

    # print(random_client_ids)
    # print(clients)

    for random_client_id in random_client_ids:
        random_clients[random_client_id] = clients[random_client_id]
    return random_clients


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


def drop_clients(clients_full: dict, clients: dict, sc_clients_full: dict, sc_clients: dict, clients_drop_frac: float):
    unused_client_ids = set(clients_full.keys()) - set(clients.keys())
    clients_retain_num = int(len(clients)*(1-clients_drop_frac))
    clients_drop_num = len(clients) - clients_retain_num
    new_client_ids = list(random.sample(list(clients.keys()), clients_retain_num)) + random.sample(unused_client_ids, clients_drop_num)
    clients = {}
    sc_clients = {}
    for new_client_id in new_client_ids:
        clients[new_client_id] = clients_full[new_client_id]
        sc_clients[new_client_id] = sc_clients_full[new_client_id]
    return clients, sc_clients


if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)

    mode = "online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning exps", mode = mode)
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

    train_dataset_size, input_channels = split_dataset(args.dataset, client_ids, args.datapoints, args.pretrained)

    print(f'Random client ids:{str(client_ids)}')
    transform=None


    print('Initializing clients...')
    for _, client in clients.items():
        (initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
    # if(args.dataset!='ham10000')
        # class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)
    print('Client Intialization complete.')
    model = importlib.import_module(f'models.{args.model}')

    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
    print('Done')

    if args.sst:
        dummy_client_id = client_ids[0]
        client_ids = client_ids[1:]
        dummy_client = clients[dummy_client_id]
        clients.pop(dummy_client_id)


    if not args.disable_dp:
        print("DP enabled")
        for _, client in clients.items():
            client.front_privacy_engine = PrivacyEngine()

    for _, client in clients.items():

        # client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)
        # client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=args.lr, momentum=0.9)
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)

    if args.sst:
        dummy_client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        dummy_client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)


    if not args.disable_dp:
        for _, client in clients.items():
            client.front_model, client.front_optimizer, client.train_DataLoader = \
                client.front_privacy_engine.make_private(
                module=client.front_model,
                data_loader=client.train_DataLoader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                optimizer=client.front_optimizer,
            )


    first_client = clients[client_ids[0]]
    num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)
    num_test_iterations = ceil(len(first_client.test_DataLoader.dataset)/args.test_batch_size)

    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_model = model.center(pretrained=args.pretrained)
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)

    if args.sst:
        dummy_client_sc = ConnectedClient(dummy_client_id, None)
        dummy_client_sc.center_model = model.center(pretrained=args.pretrained)
        dummy_client_sc.center_model.to(device)
        # dummy_client_sc.center_optimizer = optim.SGD(dummy_client_sc.center_model.parameters(), lr=args.lr, momentum=0.9)
        dummy_client_sc.center_optimizer = optim.Adam(dummy_client_sc.center_model.parameters(), args.lr)
         
 
    clients_full = clients
    sc_clients_full = sc_clients
    clients = {}
    sc_clients = {}
    # print(len(clients_full.keys()))
    # print(client_cluster_size)
    random_client_cluster_ids = list(random.sample(list(clients_full.keys()), client_cluster_size))
    for random_client_cluster_id in random_client_cluster_ids:
        clients[random_client_cluster_id] = clients_full[random_client_cluster_id]
        sc_clients[random_client_cluster_id] = sc_clients_full[random_client_cluster_id]

    st = time.time()

    for epoch in range(args.epochs):

        overall_train_acc.append(0)
        for _, client in clients.items():
            client.train_acc.append(0)
            client.iterator = iter(client.train_DataLoader)


        for iteration in range(num_iterations):
            print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
            print(clients)
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
                
            # for client_id, client in clients.items():
            #     client.remote_activations1 = copy.deepcopy(sc_clients[client_id].remote_activations1)
            #     client.backward_front()

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
                client.train_acc[-1] += client.calculate_train_acc()
        
        
        for c_id, client in clients.items():
            client.train_acc[-1] /= num_iterations
            overall_train_acc[-1] += client.train_acc[-1]

        overall_train_acc[-1] /= len(clients)
        print(f' Personalized Average Train Acc: {overall_train_acc[-1]}')

        # merge weights below uncomment 
        params = []
        for _, client in sc_clients.items():
            params.append(copy.deepcopy(client.center_model.state_dict()))
        w_glob = merge_weights(params)
        del params

        for _, client in sc_clients.items():
            client.center_model.load_state_dict(w_glob)

        params = []
        for _, client in clients.items():
            params.append(copy.deepcopy(client.back_model.state_dict()))
        w_glob_cb = merge_weights(params)
        del params

        for _, client in clients.items():
            client.back_model.load_state_dict(w_glob_cb)


        if not args.disable_dp:
            for _, client in clients.items():
                front_epsilon, front_best_alpha = client.front_privacy_engine.accountant.get_privacy_spent(delta=args.delta)
                client.front_epsilons.append(front_epsilon)
                client.front_best_alphas.append(front_best_alpha)
                print(f"([{client.id}] ε = {front_epsilon:.2f}, δ = {args.delta}) for α = {front_best_alpha}")


        if args.sst:
            dummy_client.iterator = iter(dummy_client.train_DataLoader)

            for iteration in range(num_iterations):
                print(f'\r[Server side tuning] Epoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
                dummy_client.forward_front()
                dummy_client_sc.remote_activations1 = dummy_client.remote_activations1
                dummy_client_sc.forward_center()
                dummy_client.remote_activations2 = dummy_client_sc.remote_activations2
                dummy_client.forward_back() 
                dummy_client.calculate_loss()
                dummy_client.backward_back()
                dummy_client_sc.remote_activations2.grad = dummy_client.remote_activations2.grad
                dummy_client_sc.backward_center()
                dummy_client.remote_activations1.grad = dummy_client_sc.remote_activations1.grad
                dummy_client.backward_front()
                dummy_client.step()
                dummy_client.zero_grad()
                dummy_client_sc.center_optimizer.step()
                dummy_client_sc.center_optimizer.zero_grad()
        


            # train_acc = 0
            # # average out accuracy of all random_clients
            # for _, client in random_clients.items():
            #     train_acc += client.train_acc[-1]
            # train_acc = train_acc/args.number_of_clients
            # overall_acc.append(train_acc)

        # Testing on every 10th epoch
        if epoch%10 == 0:
            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)
                for _, client in clients_full.items():
                    client.test_acc.append(0)
                    client.iterator = iter(client.test_DataLoader)
                for iteration in range(num_test_iterations):
    
                    for _, client in clients_full.items():
                        client.forward_front()

                    for client_id, client in sc_clients_full.items():
                        client.remote_activations1 = clients_full[client_id].remote_activations1
                        client.forward_center()

                    for client_id, client in clients_full.items():
                        client.remote_activations2 = sc_clients_full[client_id].remote_activations2
                        client.forward_back()

                    for _, client in clients_full.items():
                        client.test_acc[-1] += client.calculate_test_acc()

                for _, client in clients_full.items():
                    client.test_acc[-1] /= num_test_iterations
                    overall_test_acc[-1] += client.test_acc[-1]

                overall_test_acc[-1] /= len(clients_full)
                print(f' Personalized Average Test Acc: {overall_test_acc[-1]}')
            
        
            wandb.log({
                "Epoch": epoch,
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average Test Accuracy": overall_test_acc[-1],  
            })
        
        clients, sc_clients = drop_clients(clients_full, clients, sc_clients_full, sc_clients, clients_drop_frac)


    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    server side tuning: {args.sst},
                    sigma: {args.sigma}, delta: {args.delta}'''

    et = time.time()
    print(f"Time taken for this run {(et - st)/60} mins")
    wandb.log({"time taken by program in mins": (et - st)/60})

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
    # plt.savefig(f'./results/train_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
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

    if not args.disable_dp:

        X_ = first_client.front_epsilons
        Y_ = overall_test_acc
        X_Y_Spline = make_interp_spline(X_, Y_)
        X_ = np.linspace(min(X_), max(X_), 100)
        Y_ = X_Y_Spline(X_)
        ci = 0.5*np.std(Y_)/np.sqrt(len(X_))
        plt.fill_between(X_, (Y_-ci), (Y_+ci), color='blue', alpha=0.5)
        print(ci)
        plt.plot(X_, Y_)
        plt.title(f'{args.number_of_clients} Accuracy vs. Epsilon')
        plt.ylabel('Average Test Acc.')
        plt.xlabel('Epsilon')
        plt.legend()
        plt.ioff()
        plt.figtext(0.45, -0.06, plot_config, ha="center", va="center", fontsize=10)
        plt.savefig(f'./results/acc_vs_epsilon/{timestamp}.png', bbox_inches='tight')
        plt.close()
