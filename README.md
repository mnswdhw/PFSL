# priv_SLR


## 1) Build requirements:
* Python3 (>=3.8)
* pip3
* Nvidia GPU (>=12GB)

## 2)Installation
Use the following steps to install the required libraries:
* Change Directory into the project folder
* The use the command: `pip install -r requirements.txt`

## 3) Test Run

### Paramateres
The parameters options for a particular file can be checked adding -–help argument.
<br/>Optional arguments available for PFSL are:
* -h, –help show this help message and exit
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
* –-disable_wandb Disable wandb logging (default: False)
* –-pretrained Use transfer learning using a pretrained model (default: False)
* –-datapoints Number of samples of training data allotted to each client (default: 500)
* –-setting Setting you would like to run for, i.e, setting1 ,setting2 or setting4 (default: setting1)
* –-checkpoint Epoch at which personalisation phase will start (default: 50)

For reproducing the results, always add argument –-pretrained while running the PFSL script. For logging the results wandb had been used, so  to log the results to wandb, where wandb.init() is used, entity and project have to be specified according to the project folder where you wish to log the results. 
<br\>

Create a results directory in the project folder to store all the resulting plots. 

### Commands for all the scenarios

Below we state the commands for running PFSL, SL, FL, SFLv1 and SFLv2 for all the experimental scenarios.

<details> <summary><b>Setting 1: Small Sample Size (Equal), i.i.d.</b></summary>
<p> In this scenario, each client has a very small number of labelled data points ranging from 50 to 500, and all these samples are distributed identically across clients. There is no class imbalance in training data of each client. To run all the algorithms for setting 1 argument –-setting setting1 and –-datapoints [number of sample per client] has to be added. 
Rest of the arguments can be selected as per choice.

* `python PFSL_Setting124.py  -–dataset [dataset] –-setting [setting] –datapoints[number of data samples] --pretrained --model resnet18 -c 10`
* `python FL.py --dataset [dataset] --setting setting1 --datapoints [number of data samples] -c 10`
* `python SL.py --dataset [dataset] --setting setting1 --datapoints [number of data samples] -c 10`
* `python SFLv1.py --dataset [dataset] --setting setting1 --datapoints [number of data samples] -c 10`
* `python SFLv2.py --dataset [dataset] --setting setting1 --datapoints [number of data samples] -c 10`

</p></details>



<details><summary><b>Setting 2: Small Sample Size (Equal), non-i.i.d.</b></summary>
<p>In this setting, we model a situation where every client has more labelled data points from a subset of classes (prominent
classes) and less from the remaining classes. We chose to experiment with heavy label imbalance and diversity. Sample size is small and each client has equal number of training samples. To run all the algorithms for setting 2 argument --setting setting2 has to be added. For PFSL, to enable personalisation phase
from xth epoch, argument --checkpoint [x] has to be added.
Rest of the arguments can be selected as per choice.

* `python PFSL_Setting124.py --dataset [dataset] --setting setting2 --pretrained --model resnet18 -c 10`
* `python FL.py --dataset [dataset] --setting setting2 -c 10`
* `python SL.py --dataset [dataset] --setting setting2 -c 10`
* `python SFLv1.py --dataset [dataset] --setting setting2 -c 10`
* `python SFLv2.py --dataset [dataset] --setting setting2 -c 10`
  
</p>
</details>
  
<details><summary><b>Setting 3: Small Sample Size (Unequal), i.i.d.</b></summary>
<p> In this settingwe consider we there 11 clients where the Large client has 2000 labelled data points
while the other ten small clients have 150 labelled data points,
each distributed identically. The class distributions
among all the clients are the same. For evaluation purposes,
we consider a test set having 2000 data points with an identical
distribution of classes as the train set. 

To reproduce this results, run the following commands. In all the commands argument --datapoints that denotes the number of datapoints of the large client has to be added.In our case it was 2000.

* `python PFSL_Setting3.py --datapoints 2000 --dataset [dataset] --pretrained --model resnet18 -c 11`
* `python FL_Setting3.py --datapoints 2000 --dataset [dataset]  -c 11`
* `python SL_Setting3.py  --datapoints 2000 --dataset [dataset]  -c 11`
* `python SFLv1_Setting3.py --datapoints 2000 --dataset [dataset]  -c 11`
* `python SFLv2_Setting3.py --datapoints 2000 --dataset [dataset]  -c 11`
  
 </p>
 </details>


<details>
 <summary><b>Setting 4: A large number of data samples</b></summary>
<p> Here, all clients have large number of samples. This experiment was done with three different image classification datasets:
MNIST, FMNIST, and CIFAR-10. To run all the algorithms for setting 4 argument --setting setting4 has
to be added. Rest of the arguments can be selected as per choice.

* `python PFSL_Setting124.py –disable dp --dataset[dataset] --setting setting4 --pretrained --model resnet18 -c 5`
* `python FL.py --dataset [dataset] --setting setting4 -c 5`
* `python SL.py --dataset [dataset] --setting setting4 -c 5`
* `python SFLv1.py --dataset [dataset] --setting setting4 -c 5`
* `python SFLv2.py --dataset [dataset] --setting setting4 -c 5` 
</p>
</details>


<details>
 <summary><b> Setting 5: System simulation with 1000 client</b></summary>
<p> In this setting we try to simulate an environment with 1000 clients. Each client stays in the system only for 1 round which lasts only 1 epoch.
Thus, we evaluate our system for the worst possible scenario when every client cannot stay in the system for long and can only afford to make a minimal effort to participate. We assume that each client has 50 labeled data points sampled randomly but unique to the client. Within each round, we
simulate a dropout, where clients begin training but are not able to complete the weight averaging. We keep the dropout probability at 50%. 

* `python system_simulation_e2.py -c 10 --batch_size 16 --disable_dp --dataset cifar10 --model resnet18 --pretrained --epochs 100`

</p>
</details>
  
             
     


<details>
 <summary><b>Setting 6: Different Diabetic Retinopathy Datasets:</b></summary>
<p> This experiment describes the realistic scenario when healthcare centers have different sets of raw patient data for the
same disease. We have used two datasets EyePACS and APTOS whose references are given below.


<b> Dataset Sources:</b>
* Source of Dataset 1, https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
* Source of Dataset 2, https://www.kaggle.com/datasets/mariaherrerot/eyepacspreprocess

To preprocess the dataset download and store the unzipped files in data/eye_dataset1 folder and data/eye_dataset2 folder. Once verify the directory paths of the unzipped folders in the preprocess_eye_dataset_1.py and preprocess_eye_dataset_2.py files.

For Data preprocessing, run the commands mentioned below
for both the datasets <br/>
`python utils/preprocess_eye_dataset_1.py`  <br/>
`python utils/preprocess_eye_dataset 2.py`

* `python PFSL_DR.py --pretrained --model resnet18 -c 10`
* `python FL_DR.py -c 10`
* `python SL_DR.py -c 10`
* `python SFLv1_DR.py -c 10`
* `python SFLv2_DR.py -c 10`
</p>
</details>




