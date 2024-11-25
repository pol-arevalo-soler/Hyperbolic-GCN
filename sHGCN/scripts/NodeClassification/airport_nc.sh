#!/bin/bash
python3 train.py \
    --task nc \
    --dataset airport \
    --model HGCN \
    --lr 0.01 \
    --dim 16 \
    --num-layers 3 \
    --act relu \
    --bias 1 \
    --dropout 0.0 \
    --weight-decay 0.0 \
    --manifold PoincareBall \
    --log-freq 5 \
    --cuda 0 \
    --c None