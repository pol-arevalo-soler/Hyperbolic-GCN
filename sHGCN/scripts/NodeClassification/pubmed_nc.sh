#!/bin/bash

# Pre-training embeddings using the link prediction (lp) task
echo "Starting pre-training embeddings with the lp task..."
python3 train.py --task lp \
    --dataset pubmed \
    --model HGCN \
    --lr 0.01 \
    --dim 16 \
    --num-layers 2 \
    --act relu \
    --bias 1 \
    --dropout 0.4 \
    --weight-decay 0.0001 \
    --manifold PoincareBall \
    --log-freq 5 \
    --cuda 0 \
    --c None \
    --r 2.0 \
    --t 1.0 \
    --save 1 \
    --save-dir None

# Perform node classification (nc) task using pre-trained embeddings
echo "Starting node classification with the Shallow model..."
python3 train.py --task nc \
    --dataset pubmed \
    --model Shallow \
    --lr 0.01 \
    --dim 16 \
    --num-layers 3 \
    --act relu \
    --bias 1 \
    --dropout 0.2 \
    --weight-decay 0.0005 \
    --manifold Euclidean \
    --log-freq 5 \
    --cuda 0 \
    --use-feats 0 \
    --pretrained-embeddings [PATH_TO_EMBEDDINGS]