#!/bin/bash

for i in 0
do
	for j in $(seq 0 0)
	do
    	CUDA_VISIBLE_DEVICES=3 \
        python DeepGPR_TrueSim-DUE.py \
        --seed $j \
        --alpha 2.5
	done
done
