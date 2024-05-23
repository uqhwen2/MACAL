# MACAL

## Installation

```.sh
$ git clone https://github.com/uqhwen2/MACAL.git
$ conda create --name MACAL python=3.9
$ conda activate MACAL
$ pip install -r requirements.txt  # install the BMDAL baselines for benchmarking
$ pip install .  # install the Causal-Bald baselines for benchmarking
$ pip install --upgrade torch==2.1.1 torchvision==0.16.1 -f https://download.pytorch.org/whl/cu118/torch_stable.html
$ conda install protobuf==3.20.3
```
