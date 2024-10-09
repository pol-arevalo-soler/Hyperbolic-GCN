Hyperbolic Graph Convolutional Networks in PyTorch
==================================================

## 1. Overview

This repository is a graph representation learning library, containing a modified implementation of Hyperbolic Graph Convolutional Networks (HGCN) [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf) as well as  Graph Convolutional Networks (GCN) [[2]](https://arxiv.org/pdf/1609.02907.pdf).
  
All models can be trained for: 

  * Link prediction (```lp```)
  * Node classification (```nc```)

## 2. Setup

### 2.1 Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```git clone https://github.com/pol-arevalo-soler/hgcn.git```

```cd hgcn```

```conda env create -f environment.yml```

### 2.2 Installation with pip

Alternatively, if you prefer to install dependencies with pip, please follow the instructions below:

```virtualenv -p [PATH to python3.7 binary] hgcn```

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

## 4. Examples

We provide examples of training commands used to train HGCN and other graph embedding models for link prediction and node classification. In the examples below, we used a fixed random seed set to 1234 for reproducibility purposes. Note that results might slightly vary based on the machine used. To reproduce results in the paper, run each commad for 10 random seeds and average the results.

### 4.1 Training HGCN

#### Link prediction

  * Disease (Test ROC-AUC= ):

  * Airport (Test ROC-AUC=):

  * Pubmed (Test ROC-AUC= ):

  * Cora (Test ROC-AUC= ): 

```python train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

#### Node classification

  * Disease ():

  * Airport ():

To train train a HGCN node classification model on Cora and Pubmed datasets, pre-train embeddings for link prediction as decribed in the previous section. Then train a MLP classifier using the pre-trained embeddings (```embeddings.npy``` file saved in the ```save-dir``` directory). For instance for the Pubmed dataset:
 
  * PubMed ():

```python train.py --task nc --dataset pubmed --model Shallow --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0 --use-feats 0 --pretrained-embeddings [PATH_TO_EMBEDDINGS]```

  * Cora ():

### 4.2 Training GCN 

#### Link prediction 

 * Cora (): 

 * Airport ():
 
 * PubMed ():

 * Cora ():

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