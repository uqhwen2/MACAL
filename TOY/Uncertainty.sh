#!/bin/bash

for i in 0
do
	for j in $(seq 0 19)
	do
    	python ExactGPR_Uncertainty.py --seed $j
	done
done
