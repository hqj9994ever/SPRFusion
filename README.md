# SPRFusion
## Leveraging Semantic Priors for Robust Multi-Exposure Image Registration and Fusion [Under Review]
## :computer: Environment

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
