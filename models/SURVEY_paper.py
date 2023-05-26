from math import ceil
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn import ChebConv, SAGEConv, Set2Set, GraphConv, GINConv, DenseGCNConv, dense_mincut_pool
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


