#!/bin/bash

for i in 0
do
	for j in $(seq 0 0)
	do
    	CUDA_VISIBLE_DEVICES=1 \
        python causal_bald_due.py \
        --seed $j \
        --causalbald murho
	done
done