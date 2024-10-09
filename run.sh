#!/bin/bash
python3 train.py \
    --task lp \
    --dataset airport \
    --model HGCN \
    --lr 0.01 \
    --dim 16 \
    --num-layers 2 \
    --act relu \
    --bias 1 \
    --dropout 0.0 \
    --weight-decay 0.0 \
    --manifold PoincareBall \
    --log-freq 100 \
    --cuda -1 \
    --c None \
    --r 0.0 \
    --t 1.0 \