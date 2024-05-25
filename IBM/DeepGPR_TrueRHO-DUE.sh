#!/bin/bash

for i in 0
do
	for j in $(seq 0 9)
	do
        CUDA_VISIBLE_DEVICES=0 \
	python causal_bald_due.py \
 	--seed $j \
  	--causalbald rho
	done
done
