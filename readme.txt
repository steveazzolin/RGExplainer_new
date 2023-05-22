This is a Pytorch implementation of our RG-Explainer. 

You can use docker mapa17/torch_geometric to run all the scripts directly. 

If you use conda, please install:
python 3.7
pytorch 1.8.0
torch-geometric
networkx
numpy 
scikit-learn 
scipy 
pandas



## Added by Steve
python run_RG_explainer.py --paper SURVEY --dataset GRID --model CHEB --seed 42 --n_epochs 50 --size_reg 0.01 --sim_reg 1.0 --radius_penalty 0.1 > Survey_Grid_Cheb.txt 2>&1 