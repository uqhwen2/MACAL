#!/bin/bash

for i in 0
do
	for j in $(seq 0 24)
	do
    	python DeepGPR_TrueCausalBald-DUE.py --seed $j --causalbald murho
	done
done