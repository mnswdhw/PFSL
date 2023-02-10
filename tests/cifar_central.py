
def CIFAR10_iid(datapoints, path, transform_train, transform_test):

    if transform_train is None:
        #TL
        print("TL transform Cifar")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    if transform_test is None:
        #TL
        transform_test =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])


    train_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform_train,download= True)

    test_dataset = torchvision.datasets.CIFAR10(root=path,
                                            train=False,
                                            transform=transform_test, download = True)
    
    class2idx = train_dataset.class_to_idx.items()
    idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    new_train_dataset_size =  datapoints
    temp = len(train_dataset) - new_train_dataset_size

    print(len(train_dataset), new_train_dataset_size, temp)

    new_train_dataset,_ = torch.utils.data.random_split(train_dataset, (new_train_dataset_size, temp))
    new_test_dataset,_ = torch.utils.data.random_split(test_dataset, (2000, 8000))  #keeping 2k datapoints with each client


    return new_train_dataset, new_test_dataset, dict(class2idx), idx2class



def create_DataLoader(train_dataset, test_dataset, train_batch_size, test_batch_size):
    train_batch_size = train_batch_size
    test_batch_size = test_batch_size
    train_DataLoader = torch.utils.data.DataLoader(dataset= train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True)
    test_DataLoader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=True)

    return train_DataLoader, test_DataLoader



def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    

if __name__ == "__main__":

    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision import models


    train_batch_size = 256
    test_batch_size = 512
    datapoints = 2000
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    train_dataset, test_dataset,_,_ = CIFAR10_iid(datapoints, "data", None, None)
    train_loader,test_loader= create_DataLoader(train_dataset, test_dataset, train_batch_size, test_batch_size)

    model = models.resnet18(pretrained = True)
    model_children = list(model.children())
    total = len(model_children)
    fc_layer = nn.Linear(512, 10)
    model_children = model_children[:-1] + [nn.Flatten()] + [fc_layer]
    model = nn.Sequential(*model_children)

    layer_iterator = iter(model)
    for i in range(total-3):
        layer = layer_iterator.__next__()
        for param in layer.parameters():
            param.requires_grad = False

    # model = models.resnet18()
    


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', DEVICE)


    learning_rate = 0.001 #need to change/vary
    num_epochs = 150

    model_new = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    start_time = time.time()
    for epoch in range(num_epochs+1):
        
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
        print(f"Epoch {epoch} completed!")
        
        if epoch%5 == 0:
            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%% | Loss: %.3f' % (
                    epoch, num_epochs+1, 
                    compute_accuracy(model, train_loader),
                    compute_epoch_loss(model, train_loader)))

            with torch.set_grad_enabled(False): # save memory during inference
                print('Epoch: %03d/%03d | Test: %.3f%%' % (
                    epoch, num_epochs+1, 
                    compute_accuracy(model, test_loader)))
            

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))







    