# Prerequisites
- Ubuntu 16.04 environment (Python 3.6, CUDA10.0, cuDNN7)
- Install PyTorch=1.2.0 following the official instructions (https://pytorch.org)
- Install dependencies: pip install -r requirements.txt
# Data preparation
You need to download the benchmark datasets, and place them in “data/Datasets/Domain Adaptation”.

Download Link: 

GTAV: (https://download.visinf.tu-darmstadt.de/data/from_games/) 

Cityscapes: (https://www.cityscapes-dataset.com/) 

C-driving: (https://drive.google.com/drive/folders/112gXZhc_3Kxs66wWtHNY9vqsAeLX43AW)
# Evaluation

Download models and place them in “model/”.

Model link: https://pan.baidu.com/s/19Y40HUrwql0Q_Bk51IgK1g  (Access code:1234)

Please modify the settings in “experiments/GTA/test.yaml (Line21 to Line29)”, “lib/datasets/dataload.py (Line87 to Line98)”, and “tools/test_vgg.py (Line77 to Line80)” when evaluating our code on single-target and multi-target domain adaptation, respectively.

Quick start:
```
CUDA_VISIBLE_DEVICES=0 python tools/test_vgg.py --cfg experiments/GTA/test.yaml TEST.TEST_FLIP True
```
Results (w.r.t mIoU)

Source-only on Cityscapes: 27.9%

Source-only on C-Driving: 22.5%

STDA on Cityscapes: 45.5%

MTDA on C-Driving: 33.2%
