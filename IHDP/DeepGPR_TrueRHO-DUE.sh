#!/bin/bash

for i in 0
do
	for j in $(seq 0 99)
	do
    	CUDA_VISIBLE_DEVICES=0 \
        python DeepGPR_TrueCausalBald-DUE.py \
	--seed $j \
        --causalbald rho
	done
done
