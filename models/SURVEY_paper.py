from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, global_add_pool, dense_mincut_pool
from torch_geometric.nn import ChebConv, SAGEConv, Set2Set, GraphConv, GINConv, GATConv, DenseGCNConv, GATv2Conv
from torch_geometric.utils import to_dense_adj, to_dense_batch

class SURVEY_BA_2grid_Cheb(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_Cheb, self).__init__()

        self.conv1 = ChebConv(num_features, 30, K=5)
        self.conv2 = ChebConv(30, 30, K=5)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x



class SURVEY_BA_2grid_GCN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_GCN, self).__init__()

        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 30)
        self.conv3 = GCNConv(30, 30)

        self.lin1 = Linear(30, 10)
        self.lin2 = Linear(10, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        return x


class SURVEY_BA_2grid_Set2Set(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_Set2Set, self).__init__()

        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 30)
        self.conv3 = GCNConv(30, 30)
        self.set2set = Set2Set(30, 7)

        self.lin1 = Linear(60, 10)
        self.lin2 = Linear(10, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        return x


class SURVEY_BA_2grid_MinCutPool(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_MinCutPool, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        num_nodes = ceil(0.5 * 70)
        self.pool1 = Linear(32, num_nodes)

        self.conv2 = DenseGCNConv(32, 32)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(32, num_nodes)

        self.conv3 = DenseGCNConv(32, 32)

        self.lin1 = Linear(32*2, 32)
        self.lin2 = Linear(32, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, batch, edge_weights)

        x1 = embed.mean(dim=0)
        x2 = embed.sum(dim=0)
        x = torch.cat([x1, x2], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.unsqueeze(0)

    def embedding(self, x, edge_index, batch=None, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        x = x.double()
        x = self.conv1(x, edge_index).relu()
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch).double()

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)
        return x.squeeze(0)

    def graph_embedding(self, x, edge_index, edge_weights=None, batch=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, batch, edge_weights)
        x1 = embed.mean(dim=0)
        x2 = embed.sum(dim=0)
        x = torch.cat([x1, x2], dim=-1)
        return x.unsqueeze(0)


class SURVEY_BA_2grid_HO(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_HO, self).__init__()

        self.conv1 = GraphConv(num_features, 30)
        self.conv2 = GraphConv(30, 30)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x


class SURVEY_BA_2grid_GIN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_GIN, self).__init__()

        self.mlp1 = torch.nn.Linear(num_features, 30)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(30, 30)
        self.conv2 = GINConv(self.mlp2)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x


##
# BA 2 GRID HOUSE
##

class SURVEY_BA_2grid_house_Cheb(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_Cheb, self).__init__()

        self.conv1 = ChebConv(num_features, 30, K=5)
        self.conv2 = ChebConv(30, 30, K=5)
        self.conv3 = ChebConv(30, 30, K=5)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x



class SURVEY_BA_2grid_house_GCN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_GCN, self).__init__()

        self.conv1 = GCNConv(num_features, 60)
        self.conv2 = GCNConv(60, 60)
        self.conv3 = GCNConv(60, 60)
        self.conv4 = GCNConv(60, 60)

        self.lin1 = Linear(60, 60)
        self.lin2 = Linear(60, 10)
        self.lin3 = Linear(10, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        return x
    

class SURVEY_BA_2grid_house_GIN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_GIN, self).__init__()

        self.mlp1 = torch.nn.Linear(num_features, 30)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(30, 30)
        self.conv2 = GINConv(self.mlp2)

        self.lin1 = Linear(30,30)
        self.lin2 = Linear(30,num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x


class SURVEY_BA_2grid_house_Set2Set(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_Set2Set, self).__init__()

        self.conv1 = GCNConv(num_features, 60)
        self.conv2 = GCNConv(60, 60)
        self.conv3 = GCNConv(60, 60)
        self.conv4 = GCNConv(60, 60)
        self.set2set = Set2Set(60, 7)

        self.lin1 = Linear(120, 60)
        self.lin2 = Linear(60, 10)
        self.lin3 = Linear(10, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv4(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        return x


class SURVEY_BA_2grid_house_MinCutPool(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_MinCutPool, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        num_nodes = ceil(0.5 * 70)
        self.pool1 = Linear(32, num_nodes)

        self.conv2 = DenseGCNConv(32, 32)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(32, num_nodes)

        self.conv3 = DenseGCNConv(32, 32)

        self.lin1 = Linear(32*2, 32)
        self.lin2 = Linear(32, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, batch, edge_weights)

        x1 = embed.mean(dim=0)
        x2 = embed.sum(dim=0)
        x = torch.cat([x1, x2], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.unsqueeze(0)

    def embedding(self, x, edge_index, batch=None, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        x = x.double()
        x = self.conv1(x, edge_index).relu()
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch).double()

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)
        return x.squeeze(0)

    def graph_embedding(self, x, edge_index, edge_weights=None, batch=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, batch, edge_weights)
        x1 = embed.mean(dim=0)
        x2 = embed.sum(dim=0)
        x = torch.cat([x1, x2], dim=-1)
        return x.unsqueeze(0)


class SURVEY_BA_2grid_house_HO(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2):
        super(SURVEY_BA_2grid_house_HO, self).__init__()

        self.conv1 = GraphConv(num_features, 30)
        self.conv2 = GraphConv(30, 30)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        print(x.dtype)
        exit()
        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x




##
# HOUSES COLOR
##

class SURVEY_houses_color_Cheb(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_Cheb, self).__init__()

        self.conv1 = ChebConv(num_features, 30, K=5)
        self.conv2 = ChebConv(30, 30, K=5)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x



class SURVEY_houses_color_GCN(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_GCN, self).__init__()

        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 30)

        self.lin1 = Linear(30, 15)
        self.lin2 = Linear(15, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        
        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x
    

class SURVEY_houses_color_GIN(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_GIN, self).__init__()

        self.mlp1 = torch.nn.Linear(num_features, 30)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(30, 30)
        self.conv2 = GINConv(self.mlp2)

        self.lin1 = Linear(30,30)
        self.lin2 = Linear(30,num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x


class SURVEY_houses_color_Set2Set(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_Set2Set, self).__init__()

        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 30)
        self.set2set = Set2Set(30, 7)

        self.lin1 = Linear(60, 15)
        self.lin2 = Linear(15, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        return x


class SURVEY_houses_color_HO(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_HO, self).__init__()

        self.conv1 = GraphConv(num_features, 30)
        self.conv2 = GraphConv(30, 30)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x

class SURVEY_houses_color_GAT(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_GAT, self).__init__()

        self.conv1 = GATConv(num_features, 5, heads=2)
        self.conv2 = GATConv(10, 10, heads=2)
        self.conv3 = GATConv(20, 20, heads=2)

        self.lin1 = Linear(40, 10)
        self.lin2 = Linear(10, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_max_pool(embed, batch)
        return x

class SURVEY_houses_color_SAGE(torch.nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(SURVEY_houses_color_SAGE, self).__init__()

        self.conv1 = SAGEConv(num_features, 30)
        self.conv2 = SAGEConv(30, 30)
        self.conv3 = SAGEConv(30, 30)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x



##
# STARS
##

class SURVEY_stars_Cheb(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(SURVEY_stars_Cheb, self).__init__()

        self.conv1 = ChebConv(num_features, 30, K=5)
        self.conv2 = ChebConv(30, 30, K=5)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        return x



class SURVEY_stars_GCN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(SURVEY_stars_GCN, self).__init__()

        self.conv1 = GCNConv(num_features, 70)
        self.conv2 = GCNConv(70, 70)
        self.conv3 = GCNConv(70, 70)

        self.lin1 = Linear(70, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        
        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        return x
    

class SURVEY_stars_GIN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(SURVEY_stars_GIN, self).__init__()

        self.mlp1 = torch.nn.Linear(num_features, 40)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(40, 40)
        self.conv2 = GINConv(self.mlp2)

        self.lin1 = Linear(40,30)
        self.lin2 = Linear(30,num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(embed, batch)
        return x


class SURVEY_stars_Set2Set(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(SURVEY_stars_Set2Set, self).__init__()

        self.conv1 = GCNConv(num_features, 70)
        self.conv2 = GCNConv(70, 70)
        self.conv3 = GCNConv(70, 70)
        self.set2set = Set2Set(70, 7)

        self.lin1 = Linear(140, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = self.set2set(embed, batch)
        return x


class SURVEY_stars_HO(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(SURVEY_stars_HO, self).__init__()

        self.conv1 = GraphConv(num_features, 30)
        self.conv2 = GraphConv(30, 30)

        self.lin1 = Linear(30, 30)
        self.lin2 = Linear(30, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        x = self.conv1(x, edge_index, edge_weights)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weights)
        x = x.relu()
        return x

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None:  # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)

        embed = self.embedding(x, edge_index, edge_weights)
        x = global_mean_pool(embed, batch)
        return x
    
############################################################################
##
# NODE CLASSIFICATION
##
############################################################################




##
# BA-SHAPES
##

class SURVEY_shapes_Cheb(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=4):
        super(SURVEY_shapes_Cheb, self).__init__()
        #self.embedding_size = 20 * 3
        self.conv1 = ChebConv(num_features, 30, K=5)
        self.conv2 = ChebConv(30, 30, K=5)

        self.lin1 = torch.nn.Linear(30, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin1(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        x = F.relu(self.conv1(x.float(), edge_index, edge_weights))
        x = F.relu(self.conv2(x, edge_index, edge_weights))
        return x
    

class SURVEY_shapes_GCN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=4):
        super(SURVEY_shapes_GCN, self).__init__()
        #self.embedding_size = 20 * 3
        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 30)
        self.conv3 = GCNConv(30, 30)

        self.lin1 = torch.nn.Linear(30, 10)
        self.lin2 = torch.nn.Linear(10, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = F.relu(self.lin1(input_lin))
        out = self.lin2(out)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        x = self.conv1(x.float(), edge_index, edge_weights)
        x = F.relu(self.conv2(x, edge_index, edge_weights))
        x = F.relu(self.conv3(x, edge_index, edge_weights))
        return x


class SURVEY_shapes_GIN(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=4):
        super(SURVEY_shapes_GIN, self).__init__()
        #self.embedding_size = 20 * 3
        self.mlp1 = torch.nn.Linear(num_features, 70)
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = torch.nn.Linear(70, 70)
        self.conv2 = GINConv(self.mlp2)
        self.mlp3 = torch.nn.Linear(70, 70)
        self.conv3 = GINConv(self.mlp3)

        self.lin1 = torch.nn.Linear(70, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin1(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        x = F.relu(self.conv1(x.float(), edge_index, edge_weights))
        x = F.relu(self.conv2(x, edge_index, edge_weights))
        x = F.relu(self.conv3(x, edge_index, edge_weights))
        return x

class SURVEY_shapes_SAGE(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=4):
        super(SURVEY_shapes_SAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 30, aggr="sum")
        self.conv2 = SAGEConv(30, 30, aggr="sum")
        self.lin1 = Linear(30, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin1(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        x = F.relu(self.conv1(x.float(), edge_index, edge_weights))
        x = F.relu(self.conv2(x, edge_index, edge_weights))
        return x
