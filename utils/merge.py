import torch
import copy

def merge_grads(normalized_data_sizes, params):
    # params = [params_client1,
    #           params_client2,
    #           params_client3
    #           ...
    #          ]
    num_clients = len(params)
    for j,col in enumerate(zip(*params)):
        avg = 0
        for i,param in enumerate(col):
            avg += normalized_data_sizes[i] * param.grad
            # avg += param.grad

        # avg /= num_clients  # (since we are already doing weighted adding of gradients)
        for param in col:
            param.grad = copy.deepcopy(avg)
            # print("is para grad equal to average?", param.grad)

    return



def merge_weights(w):
    #after step op, merge weights 

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    return w_avg