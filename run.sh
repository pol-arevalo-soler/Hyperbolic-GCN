#!/bin/bash
python3 train.py \
    --task # choose lp or nc 
    \ --dataset # choose airport, disease_lp, disease_nc, cora, pubmed 
    \ --model # choose between GCN, HGCN or sHGCN 
    \ --lr 0.01 \
    --dim 16 \
    --num-layers 2 \
    --act relu \
    --bias 1 \
    --dropout # choose dropout 
    \ --weight-decay # choose weight-decay
    \ --manifold PoincareBall \
    --log-freq 5 \
    --cuda # -1 for cpu training or 0 for gpu training 
    \ --c # None for trainable curvature float for fixed curvature 
    \ --r 0.0 \
    --t 1.0 \