#!/bin/bash
for seed in {0..9}
do
    python3 train.py \
        --task nc \
        --dataset airport \
        --model sHGCN \
        --lr 0.01 \
        --dim 16 \
        --num-layers 2 \
        --act relu \
        --bias 1 \
        --dropout 0.0 \
        --weight-decay 0.0 \
        --manifold PoincareBall \
        --log-freq 5 \
        --cuda 0 \
        --c None \
        --r 0.0 \
        --t 1.0 \
        --seed $seed \
        --split-seed $((10 + seed))
done