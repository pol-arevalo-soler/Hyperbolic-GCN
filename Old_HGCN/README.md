# Benchmarking Hyperbolic Graph Convolutional Networks in PyTorch

## 1. Overview

This repository is a graph representation learning library, containing the original implementation of Hyperbolic Graph Convolutional Networks (HGCN) [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf)

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

### Running the HGCN Model with Attention and Local Aggregation

To run the original script [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf) for the HGCN model with attention and local aggregation, include the following arguments:

* `--use-att 1`: Enables the attention mechanism in the model.
* `--local-agg 1`: Enables local aggregation.

For example: \
```python3 train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --r 2.0 --t 1.0 --normalize-feats 0 --local-agg 1 --use-att 1```

> **Note:** This setup may run out of memory in some cases. This issue has not yet been resolved.

## 4. HGCN models

### 4.1 Training HGCN-AGG<sub>0</sub>

#### Link prediction

* Disease (TEST ROC-AUC = 81.6 $\pm$ 7.5): \
  ```python3 train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1.0 --r 2.0 --t 1.0 --normalize-feats 0```

* Airport (TEST ROC-AUC = 93.6 $\pm$ 0.4): \
  ```python3 train.py --task lp --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

* PubMed (TEST ROC-AUC = 95.1 $\pm$ 0.1): \
  ```python3 train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

* Cora (TEST ROC-AUC = 93.1 $\pm$ 0.3): \
  ```python3 train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0```

#### Node classification

* Disease (TEST F1 = 86.5 $\pm$ 6.0): \
 ```python3 train.py --task nc --dataset disease_nc --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

* Airport (TEST F1 = 85.8 $\pm$ 1.5): \
 ```python3 train.py --task nc --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.0 --weight-decay 0.0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

* Cora (TEST ACC = 76.6 $\pm$ 1.2): \
 ```python3 train.py --task nc --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold PoincareBall --log-freq 5 --cuda 0 --use-feats 0 ```

To train train a HGCN node classification model on Pubmed dataset, pre-train embeddings for link prediction as decribed above. Then train a MLP classifier using the pre-trained embeddings (```embeddings.npy``` file saved in the ```save-dir``` directory) using the Shallow model:

First pre-train embeddings using ```lp``` task: \
```python3 train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None --r 2.0 --t 1.0 --save 1 --save-dir None```


Then we can perform the Node Classification task using the ```Shallow``` model.

* PubMed (TEST ACC = 80.2 $\pm$ 0.9): \
 ```python3 train.py --task nc --dataset pubmed --model Shallow --lr 0.01 --dim 16 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0 --use-feats 0 --pretrained-embeddings [PATH_TO_EMBEDDINGS]```


### 4.3 Training HGCN-ATT<sub>0</sub>

For reproducing the 3rd model HGCN-ATT<sub>0</sub>, use HGCN-AGG<sub>0</sub> setup with the following arguments:

* `--use-att 1`: Enables the attention mechanism in the model.
* `--local-agg 0`: Disables local aggregation.