import argparse
import datetime
import hashlib
import json
import pathlib
import time
import copy
import os

import torch
from torch import optim

from components import *
from utils import *

from datasets import dataset_loaders, ground_truth_loaders
from models import model_selector
from evaluation.AUCEvaluation import AUCEvaluation
import torch_geometric as ptgeom
import networkx as nx

torch.set_num_threads(4)

def pretrain_g(g: Generator, train_comms, eval_seeds, max_size, bs, n, score_fn, dataset, writer, use_set=True):
    eval_seeds = np.array(eval_seeds)
    shuffle_index = np.arange(len(train_comms))

    for i in range(n):
        np.random.shuffle(shuffle_index)
        train_comms = train_comms[shuffle_index]
        eval_seeds = eval_seeds[shuffle_index]

        batch_loss = 0.
        for j in range(len(train_comms) // bs + 1):
            batch = train_comms[j*bs:(j+1)*bs]
            batch_seeds = eval_seeds[j*bs:(j+1)*bs]
            if len(batch) == 0:
                continue

            if dataset == "mutag":
                batch = [g.graph.sample_layerwise_expansion(x, es, max_size) for x, es in zip(batch,batch_seeds)] # for mutag
            else:
                batch = [g.graph.sample_expansion_with_high_scores(score_fn, 10, x, es, max_size) for x, es in zip(batch, batch_seeds)] # for syn1-sy4, ba

            if use_set:
                policy_loss = g.train_from_sets(batch)
            else:
                policy_loss = g.train_from_lists(batch)
            batch_loss += policy_loss
        batch_loss /= j + 1
        if use_set:
            s = 'Set '
        else:
            s = 'List'
        if writer is not None:
            writer.add_scalar(f'Pretrain/GLoss{s.strip()}', batch_loss, i)
        print(f'[Pretrain-{s} {i+1:3d}] Loss = {batch_loss:2.4f}', flush=True)


class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass


class Runner(torch.nn.Module):
    def __init__(self, args):
        super(Runner, self).__init__()

        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.args = args
        # Load Dataset
        self.graphs, self.nodefeats, ground_truth_labels, self.task = self.load_data()
        self.graphs_ori = copy.deepcopy(self.graphs)
        self.labels = copy.deepcopy(ground_truth_labels)
        print("Using", self.device)
         
        print("Num graphs = ", len(self.graphs))
        # Load pretrained models
        self.trained_model, checkpoint = model_selector.model_selector(args.paper, args.dataset, args.model, pretrained=True, return_checkpoint=True)
        # Load ground truth
        # self.explanation_labels, explain_eval_seeds = ground_truth_loaders.load_dataset_ground_truth(args.dataset)
        # Load AUC Evaluation
        # self.auc_eval= AUCEvaluation(self.task, self.explanation_labels, explain_eval_seeds)


        max_size_in_sg = 0
        if self.task == "node_task":
            self.eval_seeds = range(len(ground_truth_labels))
            print("Eval Seeds", self.eval_seeds)

            self.subgraphs = []
            for seed_node in self.eval_seeds:
                if self.args.dataset == "syn2":
                    gt_graph = ptgeom.utils.subgraph(torch.LongTensor(np.array(range(0,700,1))), torch.LongTensor(self.graphs))[0] # for syn2 to limit the ground_truth edges in the correct graph
                else:
                    gt_graph = torch.LongTensor(self.graphs)
                _, subgraph_edge, *_ = ptgeom.utils.k_hop_subgraph(seed_node, self.args.n_hop, gt_graph)
                self.subgraphs.append(subgraph_edge)

            self.pretrain_nodes = []
            self.pretrain_eval_nodes = []
            for seed_node in self.eval_seeds:
                for nl in range(self.args.n_hop-1, self.args.n_hop, 1):
                    subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(seed_node, nl+1, torch.LongTensor(self.graphs))
                    self.pretrain_nodes.append(subgraph_node.numpy())
                    self.pretrain_eval_nodes.append(seed_node)
                    if len(subgraph_node)>max_size_in_sg:
                        max_size_in_sg = len(subgraph_node)
            self.pretrain_nodes = np.array(self.pretrain_nodes)

            self.graphs = Graph(self.graphs)

            whole_graph_original_preds = self.trained_model(self.nodefeats, self.graphs.edge_index)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            print("Prediction Error for Whole Graph:", torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels)).item()/len(whole_graph_original_preds))

            subg_original_preds = [self.trained_model(self.nodefeats, self.subgraphs[i])[es] for i, es in enumerate(self.eval_seeds)]
            subg_original_preds = torch.stack(subg_original_preds)
            subg_original_preds = torch.argmax(subg_original_preds, 1)
            print("Prediction Error for Eval Seeds:", torch.sum(torch.abs(subg_original_preds-ground_truth_labels[self.eval_seeds])).item()/len(subg_original_preds))

            self.original_preds = whole_graph_original_preds
            self.original_embeds = self.trained_model.embedding(self.nodefeats, self.graphs.edge_index)            

            print("N-hop", self.args.n_hop)
            print("Max Size in SG", max_size_in_sg)

        else: # 'graph_task'
            
            self.graph_eval_seeds = range(len(self.graphs))

            each_g_actual_nodes = []
            for i in range(len(self.graphs)):
                # actual_nodes = int(torch.sum(self.nodefeats[i]))  # cunting num nodes in the graph
                actual_nodes = self.nodefeats[i].shape[0]
                each_g_actual_nodes.append(actual_nodes)
            print("total nodes", np.sum(np.array(each_g_actual_nodes)))

            each_g_actual_edges = []
            total_edges = 0
            for i in range(len(self.graphs)):
                actual_edges = []
                for e1, e2 in zip(self.graphs[i][0], self.graphs[i][1]):
                    if e1 == e2:
                        pass
                    else:
                        actual_edges.append([e1, e2])
                total_edges += len(actual_edges)
                each_g_actual_edges.append(np.array(actual_edges).T)
            print("total edges", total_edges)

            new_graphs_0 = []
            new_graphs_1 = []
            new_graphs_node = []
            cur_node_id = 0
            each_g_start_nid = []
            each_g_start_eid = []
            new_node_feats = []
            for i in range(len(self.graphs)):
                max_node_id = each_g_actual_nodes[i]

                each_g_start_nid.append(cur_node_id)
                each_g_start_eid.append(len(new_graphs_0))

                new_graphs_0.extend(each_g_actual_edges[i][0] + cur_node_id)
                new_graphs_1.extend(each_g_actual_edges[i][1] + cur_node_id)
                new_graphs_node.append(list(range(cur_node_id, cur_node_id+max_node_id)))

                cur_node_id += max_node_id
                new_node_feats.append(self.nodefeats[i][0:max_node_id])

            each_g_start_eid.append(len(new_graphs_0))
            new_graphs = np.array([new_graphs_0, new_graphs_1])
            self.graphs = Graph(new_graphs)

            self.ind_nodefeats = self.nodefeats
            self.nodefeats = torch.cat(new_node_feats) 

            # graph embedding
            print(self.ind_nodefeats[i].dtype, "whole_graph")
            whole_graph_original_preds = [self.trained_model(self.ind_nodefeats[i], self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]]-each_g_start_nid[i])[0] 
                                              for i in range(len(each_g_start_nid))]
            whole_graph_original_preds = torch.stack(whole_graph_original_preds)
            whole_graph_original_preds = torch.argmax(whole_graph_original_preds, 1)
            #ground_truth_labels = torch.argmax(ground_truth_labels, 1) Steve's GT is not one-hot
            print("Prediction Error for Graphs:", torch.sum(torch.abs(whole_graph_original_preds - ground_truth_labels)).item()/len(whole_graph_original_preds))

            g_embeds = [self.trained_model.graph_embedding(self.ind_nodefeats[i], self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]]-each_g_start_nid[i])[0] 
                                              for i in range(len(each_g_start_nid))]
            g_embeds = torch.stack(g_embeds)

            # node embedding
            n_embeds = []
            for i in range(len(each_g_start_nid)):
                temp = self.trained_model.embedding(self.ind_nodefeats[i], self.graphs.edge_index[:, each_g_start_eid[i]:each_g_start_eid[i+1]]-each_g_start_nid[i])
                n_embeds.append(temp[0:len(new_graphs_node[i])])
            n_embeds = torch.cat(n_embeds)

            # isolated nodes
            # total_isolated_nodefeats = []
            # if len(self.graphs.isolated_nodes)>0:
            #     for n_id in self.graphs.isolated_nodes:
            #         print(n_id, self.nodefeats[n_id])
            #         total_isolated_nodefeats.append(self.nodefeats[n_id])
            #     print("total", len(self.graphs.isolated_nodes), torch.sum(torch.stack(total_isolated_nodefeats),0))

            # build the graph index
            self.n_to_g_index = []
            for i in range(len(each_g_start_nid)):
                if i != len(each_g_start_nid) - 1:
                    for j in range(each_g_start_nid[i], each_g_start_nid[i+1], 1):
                        self.n_to_g_index.append(i)
                else:
                    for j in range(each_g_start_nid[i], n_embeds.size()[0], 1):
                        self.n_to_g_index.append(i)

            # ind_subgraphs
            self.ind_subgraphs = new_graphs_node
            self.each_g_start_nid = each_g_start_nid
            self.each_g_start_eid = each_g_start_eid
            self.original_preds = whole_graph_original_preds
            self.g_embeds = g_embeds
            self.max_n_nodes_in_g = np.max(np.array(each_g_actual_nodes))
            print("self.max_n_nodes_in_g",self.max_n_nodes_in_g)

        self.nodefeats_size = self.nodefeats.shape[-1]

        # Save Dir and Pretrained Dir
        self.savedir, self.pretrain_dir, self.writer = self.init_dir()
        self.args.ds_name = f'{args.dataset}-1.90'

        # Model
        self.g = self.init_g()
        if self.task == "graph_task":
            self.l = self.init_l(g_embeds, n_embeds)

    def close(self):
        self.writer.close()

    def load_data(self, shuffle=True):
        args = self.args
        # Load complete dataset
        graphs, features, ground_truth_labels, _, _, self.train_idx, self.test_idx, self.test_mask = dataset_loaders.load_dataset(args.paper, args.dataset, shuffle = shuffle)
        if isinstance(graphs, list):  # We're working with a model for graph classification
            task = "graph_task"
        else:
            task = "node_task"
        #features = torch.tensor(features)
        ground_truth_labels = torch.tensor(ground_truth_labels)
        return graphs, features, ground_truth_labels, task

    def init_dir(self):
        args = self.args
        savedir = pathlib.Path(args.savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        # writer = SummaryWriter(savedir / 'tb')
        writer = DummyWriter()
        with open(savedir / 'settings.json', 'w') as fh:
            arg_dict = vars(args)
            arg_dict['model'] = 'RGExplainer_v1'
            json.dump(arg_dict, fh, sort_keys=True, indent=4)
        pretrain_dir = pathlib.Path('pretrained')
        pretrain_dir.mkdir(exist_ok=True)
        return savedir, pretrain_dir, writer

    def init_g(self):
        args = self.args
        device = self.device
        if args.with_attr:
            g_model = Agent(args.hidden_size, args.with_attr, self.nodefeats_size).to(device)
        else:
            g_model = Agent(args.hidden_size, args.with_attr).to(device)

        g_optimizer = optim.Adam(g_model.parameters(), lr=args.g_lr)
        g = Generator(self.graphs, g_model, g_optimizer, device,
                      entropy_coef=args.entropy_coef,
                      n_rollouts=args.n_rollouts,
                      max_size=args.max_size,
                      max_reward=5.)
        if args.with_attr: 
            g.load_nodefeats(self.nodefeats)
        return g


    def init_l(self, g_embeds, n_embeds):
        args = self.args
        l = Locator(g_embeds, n_embeds, self.ind_subgraphs, args.l_lr, self.graphs.isolated_nodes, device=self.device)
        return l


    def prepare_mask(self, o_g, selected_nodes):
        n_edges = o_g.size()[1]
        n_edges_index = {}
        for i in range(n_edges):
            n_edges_index[(int(o_g[0][i]), int(o_g[1][i]))] = i

        rank = torch.zeros(n_edges)

        for i in range(len(selected_nodes)):
            for j in range(i):
                if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index and (int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                    rank[n_edges_index[(int(selected_nodes[i]), int(selected_nodes[j]))]] = self.args.max_size - i
                    rank[n_edges_index[(int(selected_nodes[j]), int(selected_nodes[i]))]] = self.args.max_size - i
                else:
                    if (int(selected_nodes[i]), int(selected_nodes[j])) in n_edges_index or (int(selected_nodes[j]), int(selected_nodes[i])) in n_edges_index:
                        print("edge pair error")

        mask = rank / (self.args.max_size-1)
        return (o_g, mask)

    def eval_loss(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]
        
        # Prediction loss
        if self.task == "node_task":
            masked_preds = [self.trained_model(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = [self.trained_model(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]
        

        masked_preds = torch.stack(masked_preds).squeeze()
        original_preds = torch.stack(original_preds)

        prediction_loss = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none') 
        prediction_loss = prediction_loss.detach().numpy()


        # Size loss
        if args.size_reg > 0:
            size_loss = args.size_reg * torch.FloatTensor([g.size()[1] for g in batch_g])
            size_loss = size_loss.detach().numpy()
        else:
            size_loss = np.array([0.0]*len(prediction_loss))


        # Radius Penalty
        if args.radius_penalty > 0:
            radius_p = args.radius_penalty * torch.FloatTensor([self.graphs.subgraph_depth(x) for x in cs])
            radius_p = radius_p.detach().numpy()
        else:
            radius_p = np.array([0.0]*len(prediction_loss))


        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = [self.trained_model.embedding(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            else:
                masked_embeds = [self.trained_model.graph_embedding(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
                sim_loss = args.sim_reg * torch.stack(sim_loss)
            sim_loss = sim_loss.detach().numpy()
        else:
            sim_loss = np.array([0.0]*len(prediction_loss))
        return prediction_loss, size_loss, sim_loss, radius_p


    def evaluate_and_print(self, prefix='', save_expl=False):
        pred_sgs = self.g.generate(self.eval_seeds)
        p_loss, s_loss, sim_loss, r_p = self.eval_loss(pred_sgs)
        pred_sgs = [x[:-1] if x[-1] == 'EOS' else x for x in pred_sgs]

        if self.args.dataset == "mutag":
            node_type = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']

            for i in self.graph_eval_seeds:
                print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]] ,"len",len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [node_type[torch.argmax(self.nodefeats[x]).item()] for x in pred_sgs[i]])
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")

            for i in self.graph_eval_seeds:
                print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]] ,"len",len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[i]])
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")

        elif self.args.dataset == "ba2":
            for i in self.graph_eval_seeds:
                print("g_id", i, "n_id", pred_sgs[i][0] - self.each_g_start_nid[self.n_to_g_index[pred_sgs[i][0]]] ,"len",len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [x - self.each_g_start_nid[self.n_to_g_index[x]] for x in pred_sgs[i]])
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")

        else: # print current first ~50 sgs for syn1-syn4
            for i in range(min(len(pred_sgs),50)):
                print("n_id", pred_sgs[i][0] ,"len",len(pred_sgs[i]), f"p_loss={p_loss[i]:.2f}", f"size_loss={s_loss[i]:.2f}", f"sim_loss={sim_loss[i]:.2f}",f"r_p={r_p[i]:.2f}", "sgs", [x - pred_sgs[i][0] for x in pred_sgs[i]])
            print(f"avg_p_loss={np.mean(p_loss):.2f}", f"avg_size_loss={np.mean(s_loss):.2f}", f"avg_sim_loss={np.mean(sim_loss):.2f}", f"r_p={np.mean(r_p):.2f}")

        if self.task == "node_task":
            explanations = [self.prepare_mask(self.subgraphs[i], x) for i, x in enumerate(pred_sgs)]
        else:
            explanations = []
            for g_id in self.graph_eval_seeds:
                explanations.append(self.prepare_mask(self.graphs.edge_index[:, self.each_g_start_eid[g_id]:self.each_g_start_eid[g_id+1]] - self.each_g_start_nid[g_id], [x - self.each_g_start_nid[g_id] for x in pred_sgs[g_id]]))

        #print(len(explanations), self.graphs.edge_index.shape)
        #print(explanations[1999])
        #print(self.graphs_ori[0])


        # Convert explanation to custom format
        if save_expl:
            task = "GraphClassification" if self.task == "graph_task" else "NodeClassification"
            PATH_to_store = (f"../Explaining-the-Explainers-in-Graph-Neural-Networks/"
                             f"Explanations/{task}/{args.dataset}/{args.model_name}/"
                             f"edge_imp/rgexpl/")

            for t in ["train", "test"]:
                for c in torch.unique(self.labels):
                    if not os.path.exists(PATH_to_store + f"{t}/{c.item()}/"):
                        os.makedirs(PATH_to_store + f"{t}/{c.item()}/")
                        print("sto creando il folder", PATH_to_store + f"{t}/{c.item()}/")

            print("\n\nSAVING EXPLANATIONS\n\n")

            for j, graph_idx in enumerate(self.train_idx):
                split = "train/"
                edge_index, weights = explanations[graph_idx]
                num_nodes = edge_index.flatten().unique().shape[0]

                data = ptgeom.data.Data(edge_index=edge_index, edge_imp=weights, num_nodes=num_nodes)
                g = nx.Graph(ptgeom.utils.to_networkx(data, edge_attrs=["edge_imp"]))

                gid = str(j) + "_" + str(self.original_preds[graph_idx].item()) + ".gpickle"
                path = PATH_to_store + split + str(self.labels[graph_idx].item()) + "/" + gid
                nx.write_gpickle(g, path)    

            for j, graph_idx in enumerate(self.test_idx):
                split = "test/"
                edge_index, weights = explanations[graph_idx]
                num_nodes = edge_index.flatten().unique().shape[0]

                data = ptgeom.data.Data(edge_index=edge_index, edge_imp=weights, num_nodes=num_nodes)
                g = nx.Graph(ptgeom.utils.to_networkx(data, edge_attrs=["edge_imp"]))

                gid = str(j) + "_" + str(self.original_preds[graph_idx].item()) + ".gpickle"
                path = PATH_to_store + split + str(self.labels[graph_idx].item()) + "/" + gid
                nx.write_gpickle(g, path)     


            #for sample_id, (edge_index, weights) in enumerate(explanations):
            #    split = "test/" if self.test_mask[sample_id] else "train/"
            #    num_nodes = edge_index.flatten().unique().shape[0]

            #   data = ptgeom.data.Data(edge_index=edge_index, edge_imp=weights, num_nodes=num_nodes)
            #    g = nx.Graph(ptgeom.utils.to_networkx(data, edge_attrs=["edge_imp"]))

                #print(nx.get_edge_attributes(g, "edge_imp"))

            #    gid = str(c[split]) + "_" + str(self.original_preds[sample_id].item()) + ".gpickle"
            #    path = PATH_to_store + split + str(self.labels[sample_id].item()) + "/" + gid
            #    nx.write_gpickle(g, path)
            #    c[split] += 1


        #auc_score = self.auc_eval.get_score(explanations)
        #print(f'[EVAL-{prefix}] AUC={auc_score}', flush=True)
        return None

    def get_g_id(self, nodes, is_check=False):
        if is_check:
            # check whether nodes belong to one graph
            flag_check = True

            try:
                for i in range(len(nodes)-1):
                    if self.n_to_g_index[nodes[i]] != self.n_to_g_index[nodes[i+1]]:
                        flag_check = False
                        break
                if flag_check == False:
                    print("Nodes do not belong to one graph!")
            except:
                print("ERRORE")
                print(len(nodes))
                print(nodes)
                print(len(self.n_to_g_index))

        return self.n_to_g_index[nodes[0]]

    def prepare_graph(self, idx, selected_nodes):
        selected_nodes = list(selected_nodes)
        if self.task == "node_task":
            return ptgeom.utils.subgraph(selected_nodes, self.graphs.edge_index)[0]
        else:
            return ptgeom.utils.subgraph(selected_nodes, self.graphs.edge_index)[0]

    def score_fn(self, cs):
        cs = [x[:-1] if x[-1] == 'EOS' else x for x in cs]
        batch_g = [self.prepare_graph(i, x) for i, x in enumerate(cs)]

        # Prediction loss
        if self.task == "node_task":
            masked_preds = [self.trained_model(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
            original_preds = [self.original_preds[x[0]] for i, x in enumerate(cs)]
        else:
            batch_g_id = [self.get_g_id(x, True) for x in cs]
            masked_preds = [self.trained_model(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
            original_preds = [self.original_preds[g_id] for g_id in batch_g_id]

        masked_preds = torch.stack(masked_preds).squeeze()
        original_preds = torch.stack(original_preds)

        v = torch.nn.functional.cross_entropy(masked_preds, original_preds, reduction='none') 

        # Size loss
        if args.size_reg > 0:
            v += args.size_reg * torch.FloatTensor([g.size()[1] for g in batch_g])

        # Raidus penalty
        if args.radius_penalty > 0:
            v += args.radius_penalty * torch.FloatTensor([self.graphs.subgraph_depth(x) for x in cs])

        # Similarity loss
        if args.sim_reg > 0:
            if self.task == "node_task":
                masked_embeds = [self.trained_model.embedding(self.nodefeats, g)[cs[i][0]] for i, g in enumerate(batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.original_embeds[x[0]]) for i, x in enumerate(cs)]
            else:
                masked_embeds = [self.trained_model.graph_embedding(self.ind_nodefeats[g_id], g - self.each_g_start_nid[g_id]) for g_id, g in zip(batch_g_id, batch_g)]
                sim_loss = [torch.norm(masked_embeds[i] - self.g_embeds[g_id]) for i, g_id in enumerate(batch_g_id)]
            v += args.sim_reg * torch.stack(sim_loss) 
        
        
        return -v.detach().numpy()


    def save(self, fname):
        if self.task == "node_task":
            data = {'g': self.g.model.state_dict()}
        else: #graph_task
            data = {'g': self.g.model.state_dict(),
                    'l': self.l.model.state_dict()}
        
        torch.save(data, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.g.model.load_state_dict(data['g'])
        if self.task == "graph_task":
            self.l.model.load_state_dict(data['l'])

    
    def simulated_return_a_subgraph(self, seeds):
        results = []
        for s in seeds:
            subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(int(s), self.args.n_hop, self.graphs.edge_index)
            results.append(list(subgraph_node.numpy()))
        return results


    def pretrain_l(self, l_epochs, sample_rate):
        self.l.train(l_epochs, self.simulated_return_a_subgraph, self.score_fn, sample_rate)
        self.eval_seeds = self.l.get_eval_seed_deterministic()
        self.evaluates_seed()



    def _pretrain(self):
        args = self.args

        if self.task == "graph_task":
            # construct pre-training samples
            self.pretrain_l(args.pretrain_l_iter, args.pretrain_l_sample_rate)
            max_size_in_sg = 0
            self.pretrain_nodes = []
            self.pretrain_eval_nodes = []
            seeds = self.l.get_eval_seed_deterministic()
            for seed_node in seeds:
                for nl in range(self.args.n_hop-1, self.args.n_hop, 1):
                    subgraph_node, _, *_ = ptgeom.utils.k_hop_subgraph(int(seed_node), nl+1, self.graphs.edge_index)
                    if len(subgraph_node)>=5:
                        self.pretrain_nodes.append(subgraph_node.numpy())
                        self.pretrain_eval_nodes.append(seed_node)
                        if len(subgraph_node)>max_size_in_sg:
                            max_size_in_sg = len(subgraph_node)

            self.pretrain_nodes = np.array(self.pretrain_nodes)
            print("pretrain_samples max_size_in_sg", max_size_in_sg)

        # pretrain g
        pretrain_g(self.g, self.pretrain_nodes, self.pretrain_eval_nodes, args.max_size, args.pretrain_g_batch_size, args.pretrain_list, self.score_fn, self.args.dataset, writer=None, use_set=False)
        pretrain_g(self.g, self.pretrain_nodes, self.pretrain_eval_nodes, args.max_size, args.pretrain_g_batch_size, args.pretrain_set, self.score_fn, self.args.dataset, writer=None, use_set=True)


    def pretrain(self):
        args = self.args
        arg_dict = vars(args)
        print(args.model)
        pretrain_related_args = ['pretrain_list', 'pretrain_set', 'pretrain_l_iter', 'pretrain_l_sample_rate',
                                'hidden_size', 'dataset', 'seed', 'max_size', 'n_hop', 'model', 'paper', 'model_name',
                                'l_lr', 'g_lr',  'pretrain_g_batch_size', 'with_attr', 'ds_name']
        code = ' '.join([str(arg_dict[k]) for k in pretrain_related_args])
        code = hashlib.md5(code.encode('utf-8')).hexdigest().upper()
        print(f'CODE: {code}')
        pth_fname = self.pretrain_dir / f'{code}.pth'
        if pth_fname.exists():
            print('Load the pre-trained model!')
            self.load(pth_fname)
        else:
            self._pretrain()
            print('Save the pre-trained model!')
            self.save(pth_fname)


    def train_g_step(self, g_it):

        shuffle_index = random.choices(list(range(len(self.eval_seeds))), k=args.g_batch_size)
        _, r, policy_loss, length = self.g.train_from_rewards(np.array(self.eval_seeds)[shuffle_index], self.score_fn)

        #self.writer.add_scalar('G/Reward', r, g_it)
        #self.writer.add_scalar('G/PolicyLoss', policy_loss, g_it)
        #self.writer.add_scalar('G/Length', length, g_it)
        print(f'Reward={r:.2f}',
              f'PLoss={policy_loss: 2.2f}',
              f'Length={length:2.1f}')


    def train_l(self, l_epochs, sample_rate):

        self.l.train(l_epochs, self.g.generate, self.score_fn, sample_rate)
        self.eval_seeds = self.l.get_eval_seed_deterministic()
        self.evaluates_seed()
    

    def evaluates_seed(self):
        
        if self.args.dataset == "ba2":
            seed_distribution = {}
            for i in range(self.max_n_nodes_in_g):
                seed_distribution[i] = 0
            for i, x in enumerate(self.eval_seeds):
                seed_distribution[x - self.each_g_start_nid[i]] += 1
            for i in range(self.max_n_nodes_in_g):
                print(i, seed_distribution[i])
            print(flush=True)

        if self.args.dataset == "mutag":
            node_type = ['C','O','Cl','H','N','F','Br','S','P','I','Na','K','Li','Ca']
            seed_feat_distribution = []
            for i, x in enumerate(self.eval_seeds):
                seed_feat_distribution.append(self.nodefeats[x])

            feat_dist = torch.mean(torch.stack(seed_feat_distribution),0)
            for i in range(len(node_type)):
                print(node_type[i],f': {feat_dist[i].item(): 2.2f}', end=" ")
            print(flush=True)

    def run(self):
        # Eval before training:
        # self.evaluate_and_print('Init')

        # Pretrain
        self.pretrain()
        if self.task == "graph_task":
            self.eval_seeds = self.l.get_eval_seed_deterministic()
            self.evaluate_and_print('Pretrained')

        # Train
        g_it = l_it = -1
        for i_epoch in range(args.n_epochs):
            print('=' * 20)
            print(f'[Epoch {i_epoch + 1:4d}]')
            tic = time.time()

            if self.task == "graph_task":
                if i_epoch==0 or i_epoch%5==0:
                    print('Update L')
                    self.train_l(args.n_l_updates, args.update_l_sample_rate)
                    l_it += 1

            print('Update G')
            self.train_g_step(g_it)
            g_it += 1

            toc = time.time()
            print(f'Elapsed Time: {toc - tic:.1f}s')

            # Eval
            #if (i_epoch + 1) % args.eval_every == 0:
            #    auc = self.evaluate_and_print(f'Epoch {i_epoch+1:4d}')
            #    metrics_string = '_'.join([f'{x * 100:0>2.0f}' for x in [auc]])
            #    self.save(self.savedir / f'{i_epoch + 1:0>5d}_{metrics_string}.pth')
                # self.writer.add_scalar('Eval/Auc', auc, i_epoch)

        now = datetime.datetime.now()
        savedir = f"trained_expl/{args.dataset}_{args.model_name}_{now.strftime('%Y%m%d%H%M%S')}.pt"
        print("Saving trained explainer to ", savedir)
        torch.save(self.state_dict(), savedir)

        self.evaluate_and_print(f'Final', save_expl=True)
        




def main(args):
    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset + Trained Model 
    parser.add_argument('--dataset', type=str, default='syn1') # syn1/syn2/syn3/syn4/ba2/mutag
    parser.add_argument('--paper', type=str, default='GNN')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=False)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--with_attr', action='store_true', default=True)
    parser.add_argument('--n_hop', type=int, default=3)
    parser.add_argument('--max_size', type=int, default=20)

    # Locator
    parser.add_argument('--pretrain_l_sample_rate', type=float, default=1.0)
    parser.add_argument('--update_l_sample_rate', type=float, default=0.2)
    parser.add_argument('--l_lr', type=float, default=1e-2)
    parser.add_argument('--pretrain_l_iter', type=int, default=200)

    # Generator
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--g_lr', type=float, default=1e-2)
    parser.add_argument('--n_rollouts', type=int, default=5)
    parser.add_argument('--pretrain_g_batch_size', type=int, default=32)
    parser.add_argument('--g_batch_size', type=int, default=128)

    # Regularization
    parser.add_argument('--size_reg', type=float, default=0.01)
    parser.add_argument('--sim_reg', type=float, default=0.01)
    parser.add_argument('--radius_penalty', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.)

    # Pretrain
    parser.add_argument('--pretrain_list', type=int, default=10) 
    parser.add_argument('--pretrain_set', type=int, default=25) 


    # coordinate 
    parser.add_argument('--n_epochs', type=int, default=10)  
    parser.add_argument('--n_g_updates', type=int, default=1)
    parser.add_argument('--n_l_updates', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)

    args = parser.parse_args()
    args.model_name = args.model
    seed_all(args.seed)

    print('= ' * 20)
    now = datetime.datetime.now()
    print(args)
    args.savedir = f'ckpts/{args.dataset}/{now.strftime("%Y%m%d%H%M%S")}/'
    print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    main(args)
    print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)


