#!/bin/bash
PYTHON=/anaconda/envs/py37_pytorch/bin/python3
for i in {0..9}
do
   python run_RG_explainer.py --dataset ba2 --seed $i --n_epochs 10 --size_reg 0.01 --sim_reg 0.01 --radius_penalty 0.01 > ba2_RG_explainer_seed_$i.txt 2>&1
done
