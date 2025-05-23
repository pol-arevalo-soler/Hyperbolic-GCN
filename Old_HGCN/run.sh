#!/bin/bash
python3 train.py \
    --task nc \
    --dataset disease_nc \
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
    --cuda -1 \
    --c None \
    --r 0.0 \
    --t 1.0 \
    --use-att 1 \
    --local-agg 1