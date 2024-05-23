#!/bin/bash

# List of .sh files to execute
files=("DeepGPR_TrueMU-DUE.sh" "DeepGPR_TrueRHO-DUE.sh")

# Loop through each file and execute them serially
for file in "${files[@]}"; do
    bash "$file"
done
