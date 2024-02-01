from yacs.config import CfgNode as CN
_C = CN()
# Base config files
_C.BASE = ['']
_C.SEED = 1
_C.init_lr = 3e-5
_C.batch_size = 128
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.f_mode = 'linear'
_C.MODEL.type = 'vit'
_C.MODEL.img_size = 224
_C.MODEL.num_classes = 4
_C.MODEL.finetune = None
# _C.MODEL.finetune = '/data/lqk/medical/GITHUB-JBHI/GITHUB-JBHI/output/malignant/vit/frt/vit_base_patch16_clip_224.laion2b/vit_base_patch16_clip_224.laion2b_best.pth'

_C.MODEL.output_dir = 'output/'
_C.MODEL.backbone = CN()
_C.MODEL.backbone.from_timm = True
# _C.MODEL.backbone.model_name = 'vit_base_patch16_clip_224.laion2b'
# _C.MODEL.backbone.model_name = 'vit_base_patch16_clip_224.openai'
_C.MODEL.backbone.model_name = 'vit_base_patch16_224.augreg_in21k'

_C.MODEL.backbone.out_dim = 768
_C.MODEL.backbone.num_patch = 196
_C.MODEL.backbone.checkpoint = False
_C.MODEL.backbone.frozen = False

_C.MODEL.FRT = CN()
_C.MODEL.FRT.centers = 300
_C.MODEL.FRT.layers = 1
_C.MODEL.FRT.mlp_ratio = 4.0
_C.MODEL.FRT.checkpoint = [False, 1]

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'Adam'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-3

def get_config():
    config = _C.clone()
    return config