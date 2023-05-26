import torch
import os

from models.GNN_paper import NodeGCN as GNN_NodeGCN
from models.GNN_paper import GraphGCN as GNN_GraphGCN
from models.PG_paper import NodeGCN as PG_NodeGCN
from models.PG_paper import GraphGCN as PG_GraphGCN
from models.SURVEY_paper import (
    SURVEY_BA_2grid_Cheb,
    SURVEY_BA_2grid_GCN,
    SURVEY_BA_2grid_Set2Set,
    SURVEY_BA_2grid_MinCutPool,
    SURVEY_BA_2grid_HO,
    SURVEY_BA_2grid_GIN,

    SURVEY_BA_2grid_house_Cheb,
    SURVEY_BA_2grid_house_GCN,
    SURVEY_BA_2grid_house_Set2Set,
    SURVEY_BA_2grid_house_MinCutPool,
    SURVEY_BA_2grid_house_HO,
    SURVEY_BA_2grid_house_GIN
)


def string_to_model(paper, dataset, model):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['syn1']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return GNN_NodeGCN(10, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "PG":
        if dataset in ['syn1']:
            return PG_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return PG_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return PG_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return PG_NodeGCN(10, 2)
        elif dataset == "ba2":
            return PG_GraphGCN(10, 2)
        elif dataset == "mutag":
            return PG_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper.lower() == "survey":
        if dataset.lower() in ['ba_2grid'] and model.lower() in ['cheb']:
            return SURVEY_BA_2grid_Cheb().double()
        elif dataset.lower() in ['ba_2grid'] and model.lower() in ['gcn']:
            return SURVEY_BA_2grid_GCN().double()
        elif dataset.lower() in ['ba_2grid'] and model.lower() in ['set2set']:
            return SURVEY_BA_2grid_Set2Set().double()
        elif dataset.lower() in ['ba_2grid'] and model.lower() in ['mincutpooling']:
            return SURVEY_BA_2grid_MinCutPool().double()
        elif dataset.lower() in ['ba_2grid'] and model.lower() in ['ho']:
            return SURVEY_BA_2grid_HO().double()
        elif dataset.lower() in ['ba_2grid'] and model.lower() in ['gin']:
            return SURVEY_BA_2grid_GIN().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['cheb']:
            return SURVEY_BA_2grid_house_Cheb().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['gcn']:
            return SURVEY_BA_2grid_house_GCN().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['set2set']:
            return SURVEY_BA_2grid_house_Set2Set().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['mincutpooling']:
            return SURVEY_BA_2grid_house_MinCutPool().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['ho']:
            return SURVEY_BA_2grid_house_HO().double()
        elif dataset.lower() in ['ba_2grid_house'] and model.lower() in ['gin']:
            return SURVEY_BA_2grid_house_GIN().double()
        else:
            print(paper, dataset, model)
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_pretrained_path(paper, dataset, model):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    if paper.upper() == "SURVEY":
        path = f"../Explaining-the-Explainers-in-Graph-Neural-Networks/models/{dataset}_{model}"
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
    return path


def model_selector(paper, dataset, model_name, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset, model_name)
    if pretrained:
        path = get_pretrained_path(paper, dataset, model_name)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        model.eval()
        # print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model
