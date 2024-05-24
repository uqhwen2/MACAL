# MACAL

## Baselines and Downstream Estimators :straight_ruler:
We acknowledge the existing contributions from the broad community to promote the open-source AI, the folders "bmdal_reg" and "causal_bald" are baselines and also estimators citing from existing literatures as follows:

BMDAL: "[Black-Box Batch Active Learning for Regression](https://arxiv.org/abs/2302.08981)":

```bibtex
@misc{kirsch2023blackbox,
    title={Black-Box Batch Active Learning for Regression},
    author={Andreas Kirsch},
    year={2023},
    eprint={2302.08981},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Causal-BALD: [Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data](https://arxiv.org/abs/2111.02275) as

```bibtex
@article{jesson2021causal,
  title={Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data},
  author={Jesson, Andrew and Tigas, Panagiotis and van Amersfoort, Joost and Kirsch, Andreas and Shalit, Uri and Gal, Yarin},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2021}
}
```


## Installation :arrow_down:

Testing Environment: 48GB NVIDIA A40 on Ubuntu 22.04 LTS platform where GPU training is enabled, otherwise the 12th Gen Intel i7-12700K 12-Core 20-Thread CPU is used.

```.sh
$ conda create --name MACAL python=3.9
$ conda activate MACAL
$ pip install -r requirements.txt  # install the BMDAL baselines for benchmarking
$ pip install .  # install the Causal-Bald baselines for benchmarking
$ pip install --upgrade torch==2.1.1 torchvision==0.16.1 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

## Started on TOY :teddy_bear: 

This toy example runs on the one-dimensional toy dataset for 25 query steps and performs 20 random seeds for averaging, the training relys on the exact Guassian process implemented by [GPyTorch](https://gpytorch.ai).

### Training Random, Uncertainty, Symmetry and Uncertainty + Symmetry (MACAL_1.0 in essence)
```.sh
$ cd TOY
$ bash Random.sh 
$ bash Uncertainty.sh
$ bash Symmetry.sh
$ bash Sim.sh
```
 ~0.5h for each of the methods to terminate for 20 seeds in total (so <2h to let all four methods to terminate). Can consider reducing the seed to 10 to save some time, the convergence behaviour won't change to much.

### Visualization on PEHE
```.sh
$ cd text_results
```
Then, run the ```plots.ipynb``` which imports all text reuslts from csv file, e.g., from truerandom/, averging 20 seeds, then save the generated figures with confidence interval under figures/.

## Example on CMNIST :framed_picture:

Empirical training time for the downstream estimator DUE-CNN on NVIDIA A40 GPU: ~2h for 50 query steps in total for one simulation.

### Training MACAL_2.5

```.sh
$ cd CMNIST
$ bash DeepGPR_TrueSim-DUE.sh  # alpha=2.5 for MACAL_2.5, can simpyly change for other regularization level for ablation stuy.
```

### Visualization on PEHE

```.sh
$ cd text_results
```
