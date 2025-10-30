# SPRFusion
## Leveraging Semantic Priors for Robust Multi-Exposure Image Registration and Fusion [Under Review]
## :memo: TODO
- :white_check_mark: Release training and testing code.
- :black_square_button: Release our training set and test set.
- :black_square_button: Release pretrained checkpoints.

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
After installing the mamba library, replace the file content of <u>mamba_ssm/ops/selective_scan_interface.py</u> with that of selective_scan_interface.py from [Vim](https://github.com/hustvl/Vim).

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
python test.py --use_align False --need_H True (if GT exists)/ False
# For dynamic scenes
python test.py --use_align True --need_H True (if GT exists)/ False
```

## :email: Contact
  If you have any other questions about the code, please open an issue in this repository or email us at  `hqj9994ever@gmail.com`.
