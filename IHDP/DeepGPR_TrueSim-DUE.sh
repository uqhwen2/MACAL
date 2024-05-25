#!/bin/bash

for i in 0
do
	for j in $(seq 0 99)
	do
    	CUDA_VISIBLE_DEVICES=3 python DeepGPR_TrueSim-DUE-MAX.py --seed $j \
        --alpha 2.5
	done
done
