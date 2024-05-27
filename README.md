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

#### Training Random, Uncertainty, Symmetry and Uncertainty + Symmetry (MACAL_1.0 in essence)
```.sh
$ cd TOY
$ bash Random.sh 
$ bash Uncertainty.sh
$ bash Symmetry.sh
$ bash Sim.sh
```
~1h for each of the methods to terminate for 20 seeds (default setting, for strictly reproducing the result as depicted in the paper) in total, so <4h to let all four methods terminate. For the sake of your valuable time, consider reducing the number of the seeds to 5 to significantly save time, the convergence behaviour won't deviate much, e.g., modify ```for j in $(seq 0 19) in Random.sh``` to be ```for j in $(seq 0 4)``` for 5 random seeds.

#### Visualization on PEHE
```.sh
$ cd text_results
```
Then, run the ```plots.ipynb``` which imports all text reuslts from csv file, e.g., from truerandom/, averging 20 seeds, then save the generated figures with confidence interval under figures/.

## Mapping Convention for the Method (in Paper) and Its Alias (in Code Script):

The following aliases are used in the shell file, e.g., ```DeepGPR_TrueSim-DUE.sh```, to call the corresponded methods:

| Method          | In Script       |
|-----------------|-----------------|
| $\mu$ BALD      | mu              |
| $\rho$ BALD     | rho             |
| $\mu\rho$ BALD  | murho           |
| QHTE            | qhte            |
| Random          | random          |
| LCMD            | lcmd            |
| BADGE           | kmeanspp        |
| BAIT            | bait            |
| MACAL_2.5       | sim_2.5         |

## Example on CMNIST :framed_picture:

Empirical training time for the downstream estimator DUE-CNN on NVIDIA A40 GPU: ~2h for 50 query steps in total for one simulation, we average over 10 random seeds for a general perfornamce.

### Operating on Multiple GPU devices:

In each of the shell file to run the method, we have ```CUDA_VISIBLE_DEVICES=0``` defaultly for cuda:0, so if there are multiple GPUs, just modify the number to the exact GPU you want to use.

#### Training MACAL_2.5

```.sh
$ cd CMNIST
$ bash DeepGPR_TrueSim-DUE.sh  # alpha=2.5 for MACAL_2.5, can simpyly change for other regularization level for ablation stuy.
```

#### Training $\mu\rho$ BALD

```.sh
$ bash DeepGPR_TrueMURHO-DUE.sh
```

#### Visualization of the Risk ($\epsilon_{\text{PEHE}}$) for 50 AL Steps

```.sh
$ cd text_results
$ python cmnist_cnn_all_plots.py
```
The figure will pop up automatically, and also saved locally under the CMNIST directory

#### Visualization of the Growing Training Set via TSNE

```.sh
$ cd embeddings
$ python tsne_plot.py --method sim_2.5  # for MACAL_2.5
$ python tsne_plot.py --method murho  # for Causal-Bald murho
$ python tsne_plot.py --method lcmd  # for BMDAL LCMD
```
Plots for 50 query steps are saved locally, e.g.,  under ```embeddings/tsne/truesim_2.5``` for MACAL_2.5.

## Example on IBM :computer:

:exclamation::exclamation::exclamation: Due to uploading file limit, to test the IBM dataset, which is a public available asset from [here](https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/blob/master/README.md). Please download the file from [the anonymous link](https://drive.google.com/drive/folders/1fKNN-IaizwpEVUuNLtsNGOI0utahN2Hr), and place the ```ibm_train.npz``` and ```ibm_test.npz``` files under the ```IBM/dataset/ibm/```:exclamation::exclamation::exclamation:

Here we average over 10 random seeds for a general perfornamce:

#### Training MACAL_2.5

```.sh
$ cd IBM
$ bash DeepGPR_TrueSim-DUE.sh  # alpha=2.5 for MACAL_2.5, can simpyly change for other regularization level for ablation stuy.
```

#### Training $\mu\rho$ BALD

```.sh
$ bash DeepGPR_TrueMURHO-DUE.sh
```

#### Visualization of the Risk ($\epsilon_{\text{PEHE}}$) for 50 AL Stpes

```.sh
$ cd text_results
$ python ibm_all_plots.py
```
The figure will pop up automatically, and also saved locally under the IBM directory

#### Visualization of the Growing Training Set via TSNE

```.sh
$ cd embeddings
$ python tsne_plot.py --method sim_2.5  # for MACAL_2.5
$ python tsne_plot.py --method murho  # for Causal-Bald murho
$ python tsne_plot.py --method lcmd  # for BMDAL LCMD
```

Plots for 50 query steps are saved locally, e.g.,  under ```embeddings/tsne/truesim_2.5``` for MACAL_2.5.

## Example on IHDP :memo:

Heads-up: Here we average over 100 random seeds for a general perfornamce. Note, that all the trained models at each query step are save locally under ```saved_models/``` folder, which can cause a stargering consumed storage for large size random seeds. So, consider moving the path to save trained model locally ```job_dir_path = Path('saved_models/IHDP/method_{}/seed_{}/step_{}'.format(config.get("acquisition_function"), seed, query_step))``` to somewhere is large enough if you wanna run through all the basedlines for 100 seeds.

#### Training MACAL_2.5

```.sh
$ cd IHDP
$ bash DeepGPR_TrueSim-DUE.sh  # alpha=2.5 for MACAL_2.5, can simpyly change for other regularization level for ablation stuy.
```

#### Training $\mu\rho$ BALD

```.sh
$ bash DeepGPR_TrueMURHO-DUE.sh
```

#### Visualization of the Risk ($\epsilon_{\text{PEHE}}$) for 47 AL Steps

```.sh
$ cd text_results
$ python ihdp_all_plots.py
```
The figure will pop up automatically, and also saved locally under the IHDP directory

#### Visualization of the Growing Training Set via TSNE

```.sh
$ cd embeddings
$ python tsne_plot.py --method sim_2.5  # for MACAL_2.5
$ python tsne_plot.py --method murho  # for Causal-Bald murho
$ python tsne_plot.py --method lcmd  # for BMDAL LCMD
```

Plots for 47 query steps are saved locally, e.g.,  under ```embeddings/tsne/truesim_2.5``` for MACAL_2.5.
