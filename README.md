# SPRFusion
## Leveraging Semantic Priors for Robust Multi-Exposure Image Registration and Fusion [Under Review]
## :memo: TODO
- :white_check_mark: Release training and testing code.
- :white_check_mark: Release our training sets and test sets.
- :white_check_mark: Release pretrained checkpoints.

## :monorail: Environment

```shell
git clone https://github.com/hqj9994ever/SPRFusion.git
conda create -n SPRFusion python=3.8.18
conda activate SPRFusion
conda install cudatoolkit==11.8 -c nvidia
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install causal_conv1d==1.1.1 # or download causal_conv1d-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl from https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.1.1 to install manually.
pip install mamba_ssm==1.1.1 # or download mamba_ssm-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl from https://github.com/state-spaces/mamba/releases/tag/v1.1.1 to install manually.
pip install -r requirements.txt
```
Note: After installing the mamba library, replace the file content of `mamba_ssm/ops/selective_scan_interface.py` with that of `selective_scan_interface.py` from [Vim](https://github.com/hustvl/Vim).

#### 📂 Data Preparation

You can download our training set and test set on [Baidu Netdisk](https://pan.baidu.com/s/18fz8JvgHQyW6pAPZNBT_Wg)(code:5pen) or [Google Drive](https://drive.google.com/drive/folders/1Xt2bGHzE5WPtI_c8lCDBn6UeB_iLHTf_?usp=sharing), or build your own training set and test set by running `get_test_data.py`.

You should put the data in the correct place in the following form.

```
SPRFusion ROOT
├─dataset 
|    ├─test # test set
|    │  ├─dynamic # dynamic scenes
|    │  │  ├─real # real dynamic scenes
|    │  │  │  ├─over
|    │  │  │  └─under
|    │  │  └─synthetic # synthetic dynamic scenes
|    │  │      ├─gt
|    │  │      ├─over
|    │  │      └─under
|    │  └─static # static scenes
|    │      ├─31nogt
|    │      │  ├─over
|    │      │  └─under
|    │      └─SICE
|    │          ├─gt
|    │          ├─over
|    │          └─under
|    └─train # training set
|        ├─flow # generated deformation fields
|        ├─trainA # under-exposed images
|        ├─trainB # over-exposed images
|        └─trainC # GT
```

#### :cookie: Pretrained Checkpoints

Download the model weights on [Baidu Netdisk](https://pan.baidu.com/s/18fz8JvgHQyW6pAPZNBT_Wg)(code:5pen) or [Google Drive](https://drive.google.com/drive/folders/1Xt2bGHzE5WPtI_c8lCDBn6UeB_iLHTf_?usp=sharing) and put it in `model_zoo/`.

## :tennis: Train
1. Pretrained models

    Before training, you need to download [SAM(ViT-B)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put it in `model_zoo/ckpt/`.

```shell
# As an example
# Pretrain registration netwrok
python train.py --task_name train0 --train_stage 'align' --num_epochs 500
# Pretrain fusion network
python train.py --task_name train0 --train_stage 'fusion' --num_epochs 500
# Joint training
python train.py --task_name train0 --train_stage 'joint' --num_epochs 1000
```

## :gun: Evaluation

```shell
# For static scenes
python test.py --input_u 'path to ue image dir' --input_o 'path to oe image dir' --gt 'path to GT dir (if GT exists)' --need_H (if GT exists)
# For dynamic scenes
python test.py --input_u 'path to ue image dir' --input_o 'path to oe image dir' --gt 'path to GT dir (if GT exists)' --use_align --need_H (if GT exists)
```

## :email: Contact
  If you have any other questions about the code, please open an issue in this repository or email us at  `hqj9994ever@gmail.com`.
