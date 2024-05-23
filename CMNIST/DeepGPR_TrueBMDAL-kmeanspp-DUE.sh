#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	CUDA_VISIBLE_DEVICES=2 python DeepGPR_TrueBMDAL-DUE.py --seed $j --bmdal kmeanspp
	done
done
