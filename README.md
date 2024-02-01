# Embracing Large Natural Data: Enhancing Medical Image Analysis via Cross-domain Fine-tuning

---
<img src=./CDTM.png>


The implementation of 'Cross-Domain Transfer Module' can be found in '/data/lqk/medical/classifer_xl/JBHI/CODE/models/modules.py'.

---
## Data Preparation
1. Download data from malignant and gastric.
    * Spanhol, F., et al. "Breast cancer histopathological database (BreakHis)." (2021).
    * Sun, Changhao, et al. "Gastric histopathology image segmentation using a hierarchical conditional random field." Biocybernetics and Biomedical Engineering 40.4 (2020): 1535-1555.

2. Check out 'Data.Preprocess.ipynb' for more detains.
    * Making CSV files
    * Data spliting
    * Downloading pretrained model weights from timm library
---
## Training
### Replace the fine-tuning method
vit & linear & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode linear --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5

vit & full & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode full --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5

vit & frt & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode full --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5
- SFT (Staged Fine-tuning)
    - firt step (freeze backbone) 
    - seconde step (Modify _C.MODEL.finetune = None in CODE/config_clip_vit.py and replace "None" with the "weight pth path" obtained after freeze training)

cnn & linear & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode conv --finetune-mode linear --csv-dir malignant_all_5fold.csv --config-name 'config_clip_convnext' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5

cnn & full & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode conv --finetune-mode full --csv-dir malignant_all_5fold.csv --config-name 'config_clip_convnext' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5

cnn & frt & BreakHis:
- python -m torch.distributed.launch --nproc_per_node=1 CODE/train.py --model-mode vit --finetune-mode frt --csv-dir malignant_all_5fold.csv --config-name 'config_clip_vit' --image-size 224 --epochs 100 --init-lr 1e-4 --batch-size 8 --num-workers 8 --nbatch_log 300 --warmup_epochs 2 --val_fold 0 --test_fold 1 --data-root ./ --gpu_id 5
- SFT (Staged Fine-tuning)
    - firt step (freeze backbone) 
    - seconde step (Modify _C.MODEL.finetune = None in CODE/config_clip_convnext.py and replace "None" with the "weight pth path" obtained after freeze training)


For HCRF dataset:
    - Modify bash "--csv-dir malignant_all_5fold.csv" to "--csv-dir gastric_all_4fold.csv" 

### Replace the natural pre-trained model
- vit: Modify "_C.MODEL.backbone.model_name = ..." in CODE/config_clip_vit.py
    - LAION-2B
        - vit_base_patch16_clip_224.laion2b
    - LAION-400M
        - vit_base_patch16_clip_224.openai
    - ImageNet-21K
        - vit_base_patch16_224.augreg_in21k

- convnext: Modify "_C.MODEL.backbone.model_name = ..." in CODE/config_clip_convnext.py
    - LAION-2B
        - convnext_base.clip_laion2b_augreg
    - LAION-400M
        - convnext_base laion400m_s13b_b51k
    - ImageNet-21K
        - convnext_base.fb_in22k
---
## Hardware & Software

Ubuntu 20.04 LTS

GPU: 2 * 3090-24G

Python: 3.9.7

Pytorch: 1.12.1+cu116

Since our server environment is a bit messy, there are many contents in the requirements.txt that do not need to be installed. You can check the version number as needed.
---
## Citation
```
@ARTICLE{10361546,
  author={Li, Qiankun and Huang, Xiaolong and Fang, Bo and Chen, Huabao and Ding, Siyuan and Liu, Xu},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Embracing Large Natural Data: Enhancing Medical Image Analysis via Cross-domain Fine-tuning}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  keywords={Biomedical imaging;Visualization;Transformers;Task analysis;Bioinformatics;Image analysis;Transfer learning;Large natural data;medical image;Cross-domain learning;Staged fine-tuning},
  doi={10.1109/JBHI.2023.3343518}}
```