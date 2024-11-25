#!/bin/bash
python3 train.py \
    --task lp \
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
    --t 1.0