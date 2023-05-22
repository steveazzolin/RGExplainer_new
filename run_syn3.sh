#!/bin/bash
PYTHON=/anaconda/envs/py37_pytorch/bin/python3
for i in {0..9}
do
   python run_RG_explainer.py --dataset syn3 --seed $i --n_epochs 30 --size_reg 0.01 --sim_reg 1.0 --radius_penalty 0.1 > syn3_RG_explainer_seed_$i.txt 2>&1
done
