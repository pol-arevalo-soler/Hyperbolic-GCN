# Simplified Hyperbolic Graph Convolutional Networks in PyTorch

## 1. Overview

This repository is a graph representation learning library, containing a modified implementation of Hyperbolic Graph Convolutional Networks (HGCN) [[1]](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf) as well as  original (HGCN) method.
  
All models can be trained for:

* Link prediction (```lp```)
* Node classification (```nc```)

## 2. Setup

### 2.1 Clone GitHub repository

This section provides step-by-step instructions on how to clone the GitHub repository for this project.

### Prerequisites

Before cloning the repository, ensure that you have the following installed on your machine:

* Git : A version control system to manage your source code. You can download and install Git from the official [Git website](https://git-scm.com/downloads).

Once Git is installed, you can proceed to clone the GitHub repository:

```git clone git@github.com:pol-arevalo-soler/Hyperbolic-GCN.git```

```cd Hyperbolic-GCN```

### 2.2 Installation with conda

### Prerequisites

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create conda environment

```conda env create -f environment.yml```

### 2.3 Installation with pip

Alternatively, if you prefer to install dependencies using `pip`, follow the steps below to set up your environment:

<!-- markdownlint-disable-file MD024 -->
### Prerequisites
<!-- markdownlint-enable -->

* Python 3.11: Ensure that Python 3.11 is installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).
* pip: `pip` usually comes pre-installed with Python. You can verify its installation by running:
  ```pip --version```

### Create virtual environment

```virtualenv -p [PATH to python3.11 binary] hgcn```

```source hgcn/bin/activate```

```pip install -r requirements.txt```

### 2.4 Project structure

This project contains two main folders:

* Old_HGCN: This folder contains the original method of Hyperbolic Graph Convolutional Networks (HGCN). You will find the implementation and relevant resources related to this method in this directory.

* sHGCN: This folder contains the new method, which is the Simplified Hyperbolic Graph Neural Networks. This implementation represents an updated approach to the concepts introduced in the original HGCN, aiming to enhance efficiency and performance.

### 2.5 Datasets

The ```data/``` folder contains source files for:

* Cora
* Pubmed
* Disease
* Airport

To run this code on new datasets, please add corresponding data processing and loading in ```load_data_nc``` and ```load_data_lp``` functions in ```utils/data_utils.py```.

## 3. Usage

### 3.1 ```set_env.sh```

Before training, open one of the 2 folders and run  ```source set_env.sh```

This will create environment variables that are used in the code.

### 3.2  ```train.py```

This script trains models for link prediction and node classification tasks.
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

## 4. Reproducibility and Examples

We provide examples of training commands used to train sHGCN (sHGCN folder) and HGCN-AGG<sub>0</sub> (Old_HGCN folder) for link prediction and node classification. To reproduce the results in this paper, run each command for 10 random seeds (from 0 to 9) and average the results. Note that our results are obtained using a GPU for these seeds, and may vary slightly based on the machine used.

For the reproducibility of the HGCN-AGG<sub>0</sub> and HGCN-ATT<sub>0</sub> models, please refer to the README in the `Old_HGCN` folder:

[README in the Old_HGCN folder](Old_HGCN/README.md)

For the new sHGCN model, please refer to the README in the `sHGCN` folder:

[README in the sHGCN folder](sHGCN/README.md)

## Citation

If you find this code useful, please cite the following paper:

## Code is based on the following repository

* [hgcn](https://github.com/HazyResearch/hgcn/tree/master)

## References

[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)
