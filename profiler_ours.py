from torchvision.models import resnet50
from thop import profile
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



model = importlib.import_module(f'models.resnet18')
model_cf = model.front(3, pretrained=True)
model_cb = model.back(pretrained = True)
model_center = model.center(pretrained=True)

input = torch.randn(64,3,224,224)
input_back = torch.randn(64, 512, 7, 7)
macs_client_CF, params_FL = profile(model_cf, inputs=(input, ))
macs_client_CB, params_SL = profile(model_cb, inputs=(input_back, ))

print(f"GFLOPS CF {((2 * macs_client_CF) / (10**9))} PARAMS: {params_FL}")
print(f"GFLOPS CB {((2 * macs_client_CB) / (10**9))} PARAMS: {params_SL}")







