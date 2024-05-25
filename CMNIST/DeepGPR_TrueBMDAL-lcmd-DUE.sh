#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	CUDA_VISIBLE_DEVICES=3 \
        python DeepGPR_TrueBMDAL-DUE.py \
        --seed $j \
        --bmdal lcmd \
        --selwithtrain True
	done
done
