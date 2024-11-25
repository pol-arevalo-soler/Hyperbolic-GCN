#!/bin/bash
python3 train.py \
    --task nc \
    --dataset cora \
    --model HGCN \
    --lr 0.01 \
    --dim 16 \
    --num-layers 2 \
    --act relu \
    --bias 1 \
    --dropout 0.2 \
    --weight-decay 0.0005 \
    --manifold PoincareBall \
    --log-freq 5 \
    --cuda 0 \
    --c None \
    --use-feats 0