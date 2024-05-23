#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
    	python DeepGPR_TrueBMDAL-DUE.py --seed $j --bmdal maxdet
	done
done
