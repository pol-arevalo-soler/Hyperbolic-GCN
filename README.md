Simplified Hyperbolic Graph Convolutional Networks in PyTorch
==================================================

## 1. Overview

This repository is a graph representation learning library, containing a modified implementation of Hyperbolic Graph Convolutional Networks (HGCN) [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf) as well as  Graph Convolutional Networks (GCN) [[2]](https://arxiv.org/pdf/1609.02907.pdf).
  
All models can be trained for: 

  * Link prediction (```lp```)
  * Node classification (```nc```)

## 2. Setup

### 2.1 Clone GitHub repository

This section provides step-by-step instructions on how to clone the GitHub repository for this project.

### Prerequisites

Before cloning the repository, ensure that you have the following installed on your machine:

- **Git**: A version control system to manage your source code. You can download and install Git from the official [Git website](https://git-scm.com/downloads).

```git clone https://github.com/pol-arevalo-soler/hgcn.git```

```cd hgcn```

### 2.2 Installation with conda

### Prerequisites

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create conda environment

```conda env create -f environment.yml```

### 2.3 Installation with pip

Alternatively, if you prefer to install dependencies using `pip`, follow the steps below to set up your environment:

### Prerequisites

- **Python 3.11**: Ensure that Python 3.11 is installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).
- **pip**: `pip` usually comes pre-installed with Python. You can verify its installation by running:
  ```pip --version```

### Create virtual environment

```virtualenv -p [PATH to python3.11 binary] hgcn```

```source hgcn/bin/activate```

```pip install -r requirements.txt```

### 2.3 Datasets

The ```data/``` folder contains source files for:

  * Cora
  * Pubmed
  * Disease 
  * Airport

To run this code on new datasets, please add corresponding data processing and loading in ```load_data_nc``` and ```load_data_lp``` functions in ```utils/data_utils.py```.

## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.2  ```train.py```

This script trains models for link prediction and node classification tasks. 
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

## 4. Reproducibility and Examples

We provide examples of training commands used to train sHGCN and HGCN-ATT<sub>0</sub> for link prediction and node classification. To reproduce the results in this paper, run each command for 10 random seeds (from 0 to 9) and average the results. Note that our results are obtained using a GPU for these seeds, and may vary slightly based on the machine used.

### Running the HGCN Model with Attention and Local Aggregation

To run the original script for the HGCN model with attention and local aggregation, include the following arguments:

- `--use-att 1`: Enables the attention mechanism in the model.
- `--local-agg 1`: Enables local aggregation.

For example: <br>
```python3 train.py --task lp --dataset disease_lp --model sHGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.2 --r 0.0 --t 1.0 --normalize-feats 0 --local-agg 1 --use-att 1```

### 4.1 Training sHGCN

#### Link prediction

  * Disease (Test ROC-AUC = 94.6 $\pm$ 0.6): <br>

  ```python3 train.py --task lp --dataset disease_lp --model sHGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.2 --r 0.0 --t 1.0 --normalize-feats 0```

  * Airport (Test ROC-AUC = 94.7 $\pm$ 0.4): <br>

  ```python3 train.py --task lp --dataset airport --model sHGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.1 --r 2.0 --t 1.0```

  * Pubmed (Test ROC-AUC = 95.7 $\pm$ 0.3): <br>
  ```python3 train.py --task lp --dataset pubmed --model sHGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

  * Cora (Test ROC-AUC = 93.4 $\pm$ 0.4): <br>
  ```python3 train.py --task lp --dataset disease_lp --model sHGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 1.0 --t 1.0```


#### Node classification

  * Disease ():

  * Airport ():

To train train a HGCN node classification model on Cora and Pubmed datasets, pre-train embeddings for link prediction as decribed in the previous section. Then train a MLP classifier using the pre-trained embeddings (```embeddings.npy``` file saved in the ```save-dir``` directory). For instance for the Pubmed dataset:
 
  * PubMed ():

```python train.py --task nc --dataset pubmed --model Shallow --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0 --use-feats 0 --pretrained-embeddings [PATH_TO_EMBEDDINGS]```

  * Cora ():

### 4.2 Training HGCN-ATT<sub>0</sub>

#### Link prediction 

 * Disease (TEST ROC-AUC = 81.6 $\pm$ 7.5): <br>
  ```python3 train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --r 2.0 --t 1.0 --normalize-feats 0```


 * Airport (TEST ROC-AUC = 93.6 $\pm$ 0.4): <br>
  ```python3 train.py --task lp --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```
 
 * PubMed (TEST ROC-AUC = 95.1 $\pm$ 0.1): <br>
  ```python3 train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

 * Cora (TEST ROC-AUC = 93.1 $\pm$ 0.3): <br>
  ```python3 train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

#### Node classification 

 * Disease (): 

 * Airport ():

 * PubMed ():

 * Cora (): 


## Citation

If you find this code useful, please cite the following paper: 


## Code is based on the following repository

 * [hgcn](https://github.com/HazyResearch/hgcn/tree/master)

## References

[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)

[2] [Kipf, T.N. and Welling, M. Semi-supervised classification with graph convolutional networks. ICLR 2017.](https://arxiv.org/pdf/1609.02907.pdf)