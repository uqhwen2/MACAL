#!/bin/bash

for i in 0
do
	for j in $(seq 0 19)
	do
    	python ExactGPR_Random.py --seed $j
	done
done
