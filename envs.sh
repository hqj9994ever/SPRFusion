conda create -n TextMEF python=3.8.18
conda activate TextMEF
conda install cudatoolkit==11.8 -c nvidia
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install causal_conv1d==1.1.1 # causal_conv1d-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.1.1 # mamba_ssm-1.1.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
# repalce the content in /root/anaconda3/envs/mamba/lib/python3.8/site-packages/mamba_ssm/ops/selective_scan_interface.py with the content in selective_scan_interface.py of vim