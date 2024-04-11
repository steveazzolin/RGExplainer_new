This is a customized Pytorch implementation of RG-Explainer for the paper [*Explaining the Explainers in Graph Neural Networks: a Comparative Study*](https://arxiv.org/abs/2210.15304). 

You can find the original code [here](https://openreview.net/forum?id=nUtLCcV24hL)





### Tip to run the code
```
python run_RG_explainer.py --paper SURVEY --dataset GRID --model CHEB --seed 42 --n_epochs 50 --size_reg 0.01 --sim_reg 1.0 --radius_penalty 0.1 > Survey_Grid_Cheb.txt 2>&1
```
