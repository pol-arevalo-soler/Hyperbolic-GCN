# Simplified Hyperbolic Graph Convolutional Networks in PyTorch

## 1. Overview

This repository is a graph representation learning library, containing a modified implementation of Hyperbolic Graph Convolutional Networks (HGCN) [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf).

All models can be trained for:

* Link prediction (```lp```)
* Node classification (```nc```)

## 2. Usage

### 2.1 ```set_env.sh```

Before training, run  ```source set_env.sh```

This will create environment variables that are used in the code.

### 2.2  ```train.py```

This script trains models for link prediction and node classification tasks.
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

## 3. Reproducibility and Examples

### 3.1 Training sHGCN

All Bash scripts required for the project are located in the `scripts` folder. 

#### Link prediction

* Disease (Test ROC-AUC = 94.6 $\pm$ 0.6): \
  ```python3 train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.2 --r 0.0 --t 1.0 --normalize-feats 0```

* Airport (Test ROC-AUC = 94.7 $\pm$ 0.4): \
  ```python3 train.py --task lp --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.1 --r 2.0 --t 1.0```

* Pubmed (Test ROC-AUC = 95.7 $\pm$ 0.3): \
  ```python3 train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

* Cora (Test ROC-AUC = 93.4 $\pm$ 0.4): \
  ```python3 train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 1.0 --t 1.0```

#### Node classification

* Disease (TEST F1 = 86.6 $\pm$ 5.8): \
  ```python3 train.py --task nc --dataset disease_nc --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

* Airport (TEST F1 = 85.5 $\pm$ 2.1): \
   ```python3 train.py --task nc --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

* Cora (TEST ACC = 76.5 $\pm$ 1.1): \
  ```python3 train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold PoincareBall --log-freq 5 --cuda 0 --use-feats 0```

To train a sHGCN node classification model on Pubmed dataset, pre-train embeddings for link prediction as decribed above. Then train a MLP classifier using the pre-trained embeddings (```embeddings.npy``` file saved in the ```save-dir``` directory) using the Shallow model:

First pre-train embeddings using lp task: \
```python3 train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0 --save 1 --save-dir None```

Then we can perform the Node Classification task using the Shallow model.

* PubMed (TEST ACC = 80.5 $\pm$ 1.1): \
  ```python3 train.py --task nc --dataset pubmed --model Shallow --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0 --use-feats 0 --pretrained-embeddings [PATH_TO_EMBEDDINGS]```