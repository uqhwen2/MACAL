#!/bin/bash

for i in 0
do
	for j in $(seq 0 24)
	do
    	python DeepGPR_TrueBMDAL-DUE.py --seed $j --bmdal kmeanspp
	done
done
