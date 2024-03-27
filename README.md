# PFSL: Personalized & Fair Split Learning with Data & Label Privacy for thin clients

## 1) Please cite as below if you use this repository:
`@software{Manas_Wadhwa_and_Gagan_Gupta_and_Ashutosh_Sahu_and_Rahul_Saini_and_Vidhi_Mittal_PFSL_2023,
author = {Manas Wadhwa and Gagan Gupta and Ashutosh Sahu and Rahul Saini and Vidhi Mittal},
month = {2},
title = {{PFSL}},
url = {https://github.com/mnswdhw/PFSL},
version = {1.0.0},
year = {2023}
}`


## 2) Credits 

To reproduce the results of the paper `Thapa, C., Chamikara, M. A., Camtepe, S., & Sun, L. (2020). SplitFed: When Federated Learning Meets Split Learning. ArXiv. https://doi.org/10.48550/arXiv.2004.12088`, we use their official source code which can be found here: https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning

For finding the FLOPs of our Pytorch split model at the client side, we use the profiler: https://github.com/Lyken17/pytorch-OpCounter


## 3) Build requirements:
* Python3 (3.8)
* pip3
* Nvidia GPU (>=12GB)
* conda


## 4)Installation
Use the following steps to install the required libraries:
* Change Directory into the project folder
* Create a conda environment using the command 
`conda create --name {env_name} python=3.8`
Eg- `conda create --name pfsl python=3.8`
* Activate conda environment using the command 
`conda activate {env_name}`
Eg- `conda activate pfsl`
* The use the command: `pip install -r requirements.txt`

## 5) Test Run

### Parameters
The parameters options for a particular file can be checked adding -–help argument.
<br/>Optional arguments available for PFSL are:
* -h, --help show this help message and exit
* -c, -–number of clients Number of Clients (default: 10)
* -b, -–batch_size Batch size (default: 128)
* –-test_batch_size Input batch size for testing (default: 128)
* -n , –-epochs Total number of epochs to train (default: 10)
* –-lr Learning rate (default: 0.001)
* -–save model Save the trained model (default: False)
* –-dataset States dataset to be used (default: cifar10)
* –-seed Random seed (default: 1234)
* –-model Model you would like to train (default: resnet18)
* –-epoch_batch Number of epochs after which next batchof clients should join (default: 5)
* –-opt_iden optional identifier of experiment (default: )
* –-pretrained Use transfer learning using a pretrained model (default: False)
* –-datapoints Number of samples of training data allotted to each client (default: 500)
* –-setting Setting you would like to run for, i.e, setting1 ,setting2 or setting4 (default: setting1)
* –-checkpoint Epoch at which personalisation phase will start (default: 50)
* --rate This arguments specifies the fraction of clients dropped off in every epoch (used in setting 5)(default: 0.5)

For reproducing the results, always add argument –-pretrained while running the PFSL script. 

Create a results directory in the project folder to store all the resulting plots using the below commands.
* `mkdir results`
* `mkdir results/FL`
* `mkdir results/SL`
* `mkdir results/SFLv1`
* `mkdir results/SFLv2`

### Commands for all the scenarios

Below we state the commands for running PFSL, SL, FL, SFLv1 and SFLv2 for all the experimental scenarios.

<details> <summary><b>Setting 1: Small Sample Size (Equal), i.i.d.</b></summary>
<p> In this scenario, each client has a very small number of labelled data points ranging from 50 to 500, and all these samples are distributed identically across clients. There is no class imbalance in training data of each client. To run all the algorithms for setting 1 argument –-setting setting1 and –-datapoints [number of sample per client] has to be added. 
Rest of the arguments can be selected as per choice. Numberof data samples can be chosen from 50, 150, 250, 350 and 500 to reproduce the results. When total data sample size was
50, batch size was chosen to be 32 and for other data samples
greater than 50 batch size was kept at 64. Test batch size was
always taken to be 512. For data sample 150, command are
given below.

* `python PFSL_Setting124.py --dataset cifar10 --setting setting1 --datapoints 150 --pretrained --model resnet18 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python FL.py --dataset cifar10 --setting setting1 --datapoints 150 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SL.py --dataset cifar10 --setting setting1 --datapoints 150 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SFLv1.py --dataset cifar10 --setting setting1 --datapoints 150 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SFLv2.py --dataset cifar10 --setting setting1 --datapoints 150 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`

</p></details>



<details><summary><b>Setting 2: Small Sample Size (Equal), non-i.i.d.</b></summary>
<p>In this setting, we model a situation where every client has more labelled data points from a subset of classes (prominent
classes) and less from the remaining classes. We chose to experiment with heavy label imbalance and diversity. Sample size is small and each client has equal number of training samples. To run all the algorithms for setting 2 argument --setting setting2 has to be added. For PFSL, to enable personalisation phase
from xth epoch, argument --checkpoint [x] has to be added.
Rest of the arguments can be selected as per choice.

* `python PFSL_Setting124.py --dataset cifar10 --model resnet18 --pretrained --setting setting2 --batch_size 64 --test_batch_size 512 --checkpoint 25 --epochs 30`
* `python FL.py --dataset cifar10 --setting setting2 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SL.py --dataset cifar10 --setting setting2 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SFLv1.py --dataset cifar10 --setting setting2 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`
* `python SFLv2.py --dataset cifar10 --setting setting2 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`


  
</p>
</details>
  
<details><summary><b>Setting 3: Small Sample Size (Unequal), i.i.d.</b></summary>
<p> In this settingwe consider we there 11 clients where the Large client has 2000 labelled data points
while the other ten small clients have 150 labelled data points,
each distributed identically. The class distributions
among all the clients are the same. For evaluation purposes,
we consider a test set having 2000 data points with an identical
distribution of classes as the train set. 

To reproduce Table IV of the paper, run setting 1 with
datapoints as 150 as illustrated above. To reproduce Table V
of the paper follow the below commands. In all the commands argument --datapoints that denotes the number of datapoints of the large client has to be added.In our case it was 2000.

* `python PFSL_Setting3.py --datapoints 2000 --dataset cifar10 --pretrained --model resnet18 -c 11 --epochs 50`
* `python SFLv1_Setting3.py --datapoints 2000 --dataset cifar10_setting3 -c 11 --epochs 100`
* `python SFLv2_Setting3.py --datapoints 2000 --dataset cifar10_setting3 -c 11 --epochs 100`
* `python FL_Setting3.py --datapoints 2000 --dataset cifar10_setting3 -c 11 --epochs 100`
* `python SL_Setting3.py --datapoints 2000 --dataset cifar10_setting3 -c 11 --epochs 100`
  
 </p>
 </details>


<details>
 <summary><b>Setting 4: A large number of data samples</b></summary>
<p> Here, all clients have large number of samples. This experiment was done with three different image classification datasets:
MNIST, FMNIST, and CIFAR-10. To run all the algorithms for setting 4 argument --setting setting4 has
to be added. Rest of the arguments can be selected as per choice. Dataset argument has 3 options: cifar10, mnist and fmnist.

* `python PFSL_Setting124.py --dataset cifar10 --setting setting4 --pretrained --model resnet18 -c 5 --epochs 20`
* `python FL.py --dataset cifar10 --setting setting4 -c 5 --epochs 20`
* `python SL.py --dataset cifar10 --setting setting4 -c 5 --epochs 20`
* `python SFLv1.py --dataset cifar10 --setting setting4 -c 5 --epochs 20`
* `python SFLv2.py --dataset cifar10 --setting setting4 -c 5 --epochs 20` 
</p>
</details>


<details>
 <summary><b> Setting 5: System simulation with 1000 client</b></summary>
<p> In this setting we try to simulate an environment with 1000 clients. Each client stays in the system only for 1 round which lasts only 1 epoch.
Thus, we evaluate our system for the worst possible scenario when every client cannot stay in the system for long and can only afford to make a minimal effort to participate. We assume that each client has 50 labeled data points sampled randomly but unique to the client. Within each round, we
simulate a dropout, where clients begin training but are not able to complete the weight averaging. We keep the dropout probability at 50%. 


Use the following command to reproduce the results: Here rate argument specifies the dropoff rate which is the numberof clients that will be dropped randomly in every epoch

* `python system_simulation_e2.py -c 10 --batch_size 16 --dataset cifar10 --model resnet18 --pretrained --epochs 100 --rate 0.3`

</p>
</details>
  
             
     


<details>
 <summary><b>Setting 6: Different Diabetic Retinopathy Datasets:</b></summary>
<p> This experiment describes the realistic scenario when healthcare centers have different sets of raw patient data for the
same disease. We have used two datasets EyePACS and APTOS whose references are given below.


<b> Dataset Sources:</b>
* Source of Dataset 1, https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
* Source of Dataset 2, https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess

To preprocess the dataset download and store the unzipped files in data/eye dataset1 folder and data/eye dataset2 folder.
For this create directories using the command:
* `mkdir data/eye_dataset1`
* `mkdir data/eye_dataset2`
<br/>
  
  
The directory structure of data is as follows:
* `data/eye_dataset1/train_images`
* `data/eye_dataset1/test_images`
* `data/eye_dataset1/test.csv`
* `data/eye_dataset1/train.csv`
* `data/eye_dataset2/eyepacs_preprocess/eyepacs_preprocess/`
* `data/eye_dataset2/trainLabels.csv`

Once verify the path of the unzipped folders in the load data function of preprocess_eye_dataset_1.py and preprocess_eye_dataset_2.py files.

For Data preprocessing, run the commands mentioned below
for both the datasets <br/>
`python utils/preprocess_eye_dataset_1.py`  <br/>
`python utils/preprocess_eye_dataset 2.py`

* `python PFSL_DR.py --pretrained --model resnet18 -c 10 --batch_size 64 --test_batch_size 512 --epochs 50`
* `python FL_DR.py -c 10 --batch_size 64 --test_batch_size 512 --epochs 50`
* `python SL_DR.py --batch_size 64 --test_batch_size 512 --epochs 50`
* `python SFLv1_DR.py --batch_size 64 --test_batch_size 512 --epochs 50`
* `python SFLv2_DR.py --batch_size 64 --test_batch_size 512 --epochs 50`
</p>
</details>


## (6) Test Example Outputs for different Settings

### Setting 1

Command: `python PFSL_Setting124.py --dataset cifar10 --setting setting1 --datapoints 150 --pretrained --model resnet18 -c 10 --batch_size 64 --test_batch_size 512 --epochs 100`

Maximum test accuracy and time taken for the run is noted in this setting. 

Final Output of the above command is as follows: <br/>
* Epoch: 100, Iteration: 3/3
* Training Accuracy:  100.0
* Maximum Test Accuracy:  82.52188846982759
* Time taken for this run 49.36871874332428 mins

### Setting 2

Command: `python PFSL_Setting124.py --dataset cifar10 --model resnet18 --pretrained --setting setting2 --batch_size 64 --test_batch_size 512 --checkpoint 25 --epochs 30`

After the 25th layer, personalization phase begins since checkpoint is specified as 25 in the above command. It outputs F1 Score just before the start of the personalization phase and the maximum F1 Score achieved in that phase. 

Final Output: <br/>
* Epoch: 25, Iteration: 8/8freezing the center model
* Epoch: 26, Iteration: 8/8freezing the center model
* F1 Score at epoch  25  :  0.8151361976947273
* Epoch: 30, Iteration: 8/8
* Training Accuracy:  100.0
* Maximum F1 Score:  0.9509444261471961
* Time taken for this run 11.245620834827424 mins


### Setting 3

Command: `python PFSL_Setting3.py --datapoints 2000 --dataset cifar10 --pretrained --model resnet18 -c 11 --epochs 50`
<br/>

This command will print statements of the form as below every epoch<br/>
* Large client train/Test accuracy xx.xx
* Epoch 19 C1-C10 Average Train/Test Acc: xx.xx

Final Output will be the maximum test accuracy of the large client and the maximum average test accuracy of the remaining clients which for the above command is <br/>
* Average C1 - C10 test accuracy:  86.1162109375
* Large Client test Accuracy:  87.109375
* Time taken for this run 7.857871949672699 mins


### Setting 4

Command: `python PFSL_Setting124.py --dataset cifar10 --setting setting4 --pretrained --model resnet18 -c 5 --epochs 20`
<br/>

Final Output of the above command is as follows <br/>
* Epoch: 20, Iteration: 79/79
* Training Accuracy:  98.90427215189872
* Maximum Test Accuracy:  94.1484375
* Time taken for this run 36.06000682512919 mins

### Setting 5

Command: `python system_simulation_e2.py -c 10 --batch_size 16 --dataset cifar10 --model resnet18 --pretrained --epochs 40 --rate 0.3`

For every epoch it prints the average train accuracy and the number of clients that are dropped off. Next, it prints the ids of the clients that are not dropped off. 


Final Output of the above command is as follows <br/>
* Personalized Average Test Acc: 88.19791666666666
* Time taken for this run 18.778587651252746 mins 




### Setting 6

Command: `python PFSL_DR.py --pretrained --model resnet18 -c 10 --batch_size 64 --test_batch_size 512 --epochs 50`

Average test accuracy for clients having the APTOS dataset and clients having the EyePACS dataset is noted separately. Also, F1 Score for one representative client from each group is noted. The command above outputs these metrics for the epoch in which the maximum average test accuracy of all the clients is achieved. 

Final Output: <br/>
* Epoch: 50, Iteration: 8/8
* Time taken for this run 75.75171089967093 mins
* Time taken for this run 75.75171089967093 mins
* Maximum Personalized Average Test Acc: 79.24868724385246  
* Maximum Personalized Average Train Acc: 97.85606971153845  
* Client0 F1 Scores: 0.772644561137067
* Client5 F1 Scores:0.5906352306590767
* Personalized Average Test Accuracy for Clients 0 to 4 ": 85.21932633196721
* Personalized Average Test Accuracy for Clients 5 to 9": 73.27804815573771




## (7) Quick Validation of Environment 

Command: `python PFSL_Setting124.py --dataset cifar10 --setting setting1 --datapoints 50 --pretrained --model resnet18 -c 5 --batch_size 64 --test_batch_size 512 --epochs 2`

Output <br/>

* Training Accuracy:  82.0
* Maximum Test Accuracy:  57.88355334051723
* Time taken for this run 0.4888111670811971 mins 












