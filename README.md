# mutDDG-SSM

****
- [Overview](#Overview)
- [Functions](#Functions)
- [System_Requirements](#System_Requirements)
- [Installation_Guide](#Installation_Guide)
- [Usage_and_Demo](#Usage_and_Demo)
****
## Overview
Repository for the application of mutDDG-SSM.
Thermostability is a fundamental property of proteins to maintain their biological functions. Predicting protein stability changes upon mutation is important for our understanding protein structure-function relationship, and is also of great interest in protein engineering and pharmaceutical design. Here we present mutDDG-SSM, a deep learning-based framework that uses the geometric representations encoded in protein structure to predict the mutation-induced protein stability changes.

![image](https://github.com/SJGLAB/mutDDG_SSM/assets/115686053/70825180-8bc8-4fcc-9a95-cc54999508ec)



## Functions
***[10fold_predict.py](10fold_predict.py)***  To obtain the results of test sets.  
***[predict_single.py](predict_single.py)***  To obtain predicted ΔΔG of an example.   
***[data_process_batch_multi.py](data_process_batch_multi.py)***  preprocessing codes.        
***[models36_msd.py](models36_msd.py)***  codes for GAT model.        
***[all_util_monomer_batch36_msd.py](all_util_monomer_batch36_msd.py)***  utils for training, including feature building, loss computation and others.     

## System_Requirements
### **1. Hardware requirements**  

Only a standard computer with enough RAM to support the in-memory operations is required.  
A GPU equipped is better. 

### **2. Software requirements**  

#### OS requirements  
The codes are tested on the following OSes:   
- Linux x64  
- Windows 10 x64

And the following x86_64 version of Python:  
- Python 3.8
  
#### Python dependencies   
- numpy   
- pandas
- Biopython
- sys  
- math
- xgboost
- scipy
- joblib
- scikit-learn
- torch_geometric
- Pytorch

#### Software dependencies 
- [Rosetta](https://www.rosettacommons.org/software/)

## Installation_Guide
### Download the codes
```
git clone https://github.com/SJGLAB/mutDDG_SSM.git
```
### Prepare the environment
We recommand you to use [Anaconda](https://www.anaconda.com/) to prepare the environments.  
You can create the environment via  
```
conda create -n mutDDG_SSM python=3.8
conda activate mutDDG_SSM
pip install torch numpy pandas biopython math xgboost scipy joblib scikit-learn
conda install -c pyg pyg
```
The installation takes about 20 mins. 

## Usage_and_Demo
* To run an example on 1AMQ
```
python3 predict_single.py
```





