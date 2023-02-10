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
import server
import multiprocessing
from sklearn.metrics import classification_report
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
# from utils import dataset_settings, datasets,
from utils import get_eye_dataset
import torch.nn.functional as F
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


def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    client.load_data(args.dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


def select_random_clients(clients):
    random_clients = {}
    client_ids = list(clients.keys())
    random_index = random.randint(0,len(client_ids)-1)
    random_client_ids = client_ids[random_index]

    print(random_client_ids)
    print(clients)

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
    plt.savefig('plot_setting_DR_exp.png')

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

    mode = "online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning_exps_vidhi", mode = mode)
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
    overall_test_acc1 = []
    overall_test_acc2 = []


    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    # train_dataset_size, input_channels = split_dataset_cifar10tl_exp(client_ids, args.datapoints)

    print(f'Random client ids:{str(client_ids)}')
    transform=None
    max_epoch=0
    max_f1=0


    print('Initializing clients...')
    
  

    d1, d2=get_eye_dataset.get_idxs()




    input_channels=3
 



    i=0
    dict_user_train=dict()
    dict_user_test_personalized=dict()
    client_idxs=dict()

    for _, client in clients.items():
        dict_user_train[_]=d1[i]
        dict_user_test_personalized[_]=d2[i]
        # dict_user_test_generalized[_]=dict_users_test_equal[i%2]
        client_idxs[_]=i
        i+=1
    for _, client in clients.items():
        # client.train_dataset=DatasetSplit(train_full_dataset, dict_user_train[_])
        client_idx=client_idxs[_]
        if(client_idx>=0 and client_idx<=4 ):
            train_data_id=1
            test_data_id=3
        elif(client_idx>=5 and client_idx<=9):
            train_data_id=2
            test_data_id=4
        
        client.train_dataset=get_eye_dataset.get_eye_data(dict_user_train[_], train_data_id)
        client.test_dataset_personalised=get_eye_dataset.get_eye_data(dict_user_test_personalized[_], test_data_id)
        client.create_DataLoader(args.batch_size, args.test_batch_size)



    # for _, client in clients.items():
        # (initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
    # if(args.dataset!='ham10000'):
    #     class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)
    print('Client Intialization complete.')
    model = importlib.import_module(f'models.{args.model}')
    count=0
    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
        if(count==0):
            print(f'front model:\n {client.front_model}')
            print(f'back model:\n {client.back_model}')
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
    # num_test_iterations_generalization = ceil(len(first_client.test_DataLoader_generalized.dataset)/args.test_batch_size)
    num_test_iterations_personalization = ceil(len(first_client.test_DataLoader_personalised.dataset)/args.test_batch_size)
    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)
    
    count=0
    for _,s_client in sc_clients.items():
        s_client.center_model = model.center(pretrained=args.pretrained)
        count+=1
        if(count==1):
            print(f'center model: {s_client.center_model}')
        
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)

    if args.sst:
        dummy_client_sc = ConnectedClient(dummy_client_id, None)
        dummy_client_sc.center_model = model.center(pretrained=args.pretrained)
        dummy_client_sc.center_model.to(device)
        # dummy_client_sc.center_optimizer = optim.SGD(dummy_client_sc.center_model.parameters(), lr=args.lr, momentum=0.9)
        dummy_client_sc.center_optimizer = optim.Adam(dummy_client_sc.center_model.parameters(), args.lr)
         

    st = time.time()

    macro_avg_f1_3classes=[]
    macro_avg_f1_dict={}

    criterion=F.cross_entropy
    # for _, client in clients.items(): 
    #     wandb.watch(client.front_model, criterion, log="all",log_freq=2) 
    #     wandb.watch(client.back_model, criterion, log="all", log_freq=2)
    # for _, s_client in sc_clients.items():
    #     wandb.watch(s_client.center_model, criterion, log="all", log_freq=2)

    for epoch in range(args.epochs):
        # if(epoch==args.checkpoint):
            # for _, s_client in sc_clients.items():
                # s_client.center_model.freeze(epoch, pretrained=True)

        overall_train_acc.append(0)
        for _, client in clients.items():
            client.train_acc.append(0)
            client.iterator = iter(client.train_DataLoader)
            
            
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

        for _, client in sc_clients.items():
            client.center_model.load_state_dict(w_glob)

        params = []
        # if(epoch <=args.checkpoint):
        for _, client in clients.items():
            params.append(copy.deepcopy(client.back_model.state_dict()))
        w_glob_cb = merge_weights(params)

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

        # Testing on every 5th epoch
        
        if (epoch%1 == 0 ):

            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)
                overall_test_acc1.append(0)
                overall_test_acc2.append(0)

                # for 
                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.iterator = iter(client.test_DataLoader_personalised)
                    client.pred=[]
                    client.y=[]
                for iteration in range(num_test_iterations_personalization):
    
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
                    client.test_acc[-1] /= num_test_iterations_personalization
                    overall_test_acc[-1] += client.test_acc[-1]
                    idx=client_idxs[_]
                    if(idx>=0 and idx<5):
                        overall_test_acc1[-1] += client.test_acc[-1]
                    elif(idx>=5 and idx<10):
                        overall_test_acc2[-1] += client.test_acc[-1]
               
                    clr=classification_report(np.array(client.y), np.array(client.pred), output_dict=True)
                    print(clr)
                    
                    curr_f1=(clr['0']['f1-score']+clr['1']['f1-score']+clr['2']['f1-score'])/3
                    macro_avg_f1_3classes.append(curr_f1)
                    macro_avg_f1_dict[idx]=curr_f1
                    if(epoch==0):
                        print("client ", idx)
                        print(classification_report(np.array(client.y), np.array(client.pred)))
                        print(macro_avg_f1_3classes)
                
                overall_test_acc[-1] /= len(clients)
                overall_test_acc1[-1] /=5
                overall_test_acc2[-1] /= 5

                f1_avg_all_user=sum(macro_avg_f1_3classes)/len(macro_avg_f1_3classes)
                macro_avg_f1_3classes=[]
                # macro_avg_f1_dict={}
                print(f' Personalized Average Test Acc: {overall_test_acc[-1]}  ')
               
        
            wandb.log({
                "Epoch": epoch,
                "Client0_F1 Scores": macro_avg_f1_dict[0],
                # "Client1_F1_Scores":macro_avg_f1_dict[1],
                "Client5_F1_Scores":macro_avg_f1_dict[5],
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average Test Accuracy": overall_test_acc[-1],  
                "Personalized Average Test Accuracy 1": overall_test_acc1[-1],
                "Personalized Average Test Accuracy 2": overall_test_acc2[-1],

            })
            macro_avg_f1_dict={}

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


