# Embracing Large Natural Data: Enhancing Medical Image Analysis via Cross-domain Fine-tuning


- ### Overview of our CDTM
<img src=./Figures/overflow.png>

## 1.HARDWARE & SOFTWARE

Ubuntu 20.04

CPU: 12700k

GPU: 1 * 3090, 24G

Python: 3.8.0

Pytorch: 1.13.1+cu113

## 2.Installation
### Clone this repository

```Shell
git clone https://github.com/qklee-lz/CDTM.git
```


### Create a conda virtual environment
```Shell
conda create -n CDTM python=3.8 -y
conda activate CDTM
```
### Install dependencies
```Shell
# install pytorch
pip install torch==1.13.0 torchvision==0.12.0
# install python packages
python setup.py build develop
```

## 3.Data preparation
### BreakHis
- Spanhol, F., et al. "Breast cancer histopathological database (BreakHis)." (2021).



### HCRF
- Sun, Changhao, et al. "Gastric histopathology image segmentation using a hierarchical conditional random field." Biocybernetics and Biomedical Engineering 40.4 (2020): 1535-1555.



## 4.Codes
Codes will be released after paper acceptance.
