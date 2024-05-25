#!/bin/bash

for i in 0
do
	for j in $(seq 0 99)
	do
    	python DeepGPR_TrueBMDAL-DUE.py --seed $j --bmdal bait
	done
done
