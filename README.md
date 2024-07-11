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
***[run_mutDDG-SSM.sh](run_mutDDG-SSM.sh)***  Executable file to run predictions.  
***[run_mutation.py](run_mutation.py)***  Help to generate mutant protein, you can your own mutant structure.
## Codes
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

And the following x86_64 version of Python:  
- Python 3.8
  
#### Python dependencies   
- The codes are packaged.

#### Software dependencies 
- [Rosetta](https://www.rosettacommons.org/software/)
- [DSSP](http://swift.cmbi.ru.nl/gv/dssp/)

## Installation_Guide
### Download the codes and unzip
```
git clone https://github.com/SJGLAB/mutDDG_SSM.git
cd mutDDG_SSM
unzip dist.zip
unzip build.zip
```

## Usage_and_Demo
* To run predictions
```
sh run_mutDDG-SSM.sh #You can input the wild type and mutant proteins with information
```







