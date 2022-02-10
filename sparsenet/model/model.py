# Created at 2020-04-10
# Summary: graph encoders

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import Set2Set, MessagePassing, global_add_pool, global_mean_pool, global_max_pool, \
    GlobalAttention
from torch_geometric.utils import add_self_loops

from sparsenet.util.util import summary, fix_seed, random_pygeo_graph


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_feat_dim, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.edge_feat_dim = edge_feat_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding = torch.nn.Linear(self.edge_feat_dim, emb_dim)  # torch.nn.Embedding(num_bond_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), self.edge_feat_dim)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)  # LongTensor of shape [32, 1]

        edge_attr_ = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_attr_ = edge_attr_.type(torch.FloatTensor).to(edge_attr.device)
        # summary(edge_attr, 'edge_attr after adding self loop')

        edge_embeddings = self.edge_embedding(edge_attr_)

        return self.propagate(edge_index, size=[x.size(0), x.size(0)], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, node_feat_dim, edge_feat_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        if self.num_layer < 2:
            print("Warning: Number of GNN layers must be greater than 1.")

        ########
        self.x_embedding0 = torch.nn.Linear(self.node_feat_dim, emb_dim)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(self.edge_feat_dim, emb_dim, aggr="add"))
            else:
                NotImplementedError

        ### List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding0(x)  # self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    # @profile
    def __init__(self, num_layer, emb_dim, node_feat_dim, edge_feat_dim, num_tasks, JK="last", drop_ratio=0,
                 graph_pooling="mean", gnn_type="gin", force_pos=False, mlp=False):
        """

        :param num_layer:
        :param emb_dim:
        :param node_feat_dim:
        :param edge_feat_dim:
        :param num_tasks:
        :param JK:
        :param drop_ratio:
        :param graph_pooling:
        :param gnn_type:
        :param force_pos: force postive. If true, add non-linear layer in the end.
        """
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.edge_feat_dim = edge_feat_dim
        self.force_pos = force_pos
        self.mlp = mlp

        if self.num_layer < 2:
            print("Warning: Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, node_feat_dim, edge_feat_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

        if self.mlp:
            self.graph_pred_linear = torch.nn.Sequential(
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mult * self.emb_dim, self.mult * self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks))

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio) # important
        self.gnn.load_state_dict(torch.load(model_file))

    # @timefunc
    def forward(self, *argv, ini=False):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        rep = self.graph_pred_linear(self.pool(node_representation, batch))
        if ini and len(argv) == 1:
            ini_tsr = torch.stack([data.ini] * rep.size(1), dim=1)

        if self.force_pos:
            if ini:
                # important: new version. has not tested it. Does it works well for amazons?
                return torch.nn.ReLU()(rep + ini_tsr) + 1
                # return 0.5 * rep + torch.nn.ReLU()(ini_tsr) # + torch.zeros(rep.size()).to(rep.device)
            else:
                return 1 + torch.nn.ReLU()(rep)  # important: add 1 by default. not sure it's the best.
        else:
            return rep


parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--gnn_type', type=str, default='gin', help='')

if __name__ == "__main__":
    fix_seed()
    edge_feat_dim = 1
    node_feat_dim = 5
    n_node, n_edge = 320, 5000
    n_layer = 3
    emb_dim, out_dim = 50, 18

    model = GNN_graphpred(n_layer, emb_dim, node_feat_dim, edge_feat_dim, out_dim, mlp=False)

    g1 = random_pygeo_graph(n_node, node_feat_dim, n_edge, edge_feat_dim, device='cpu')
    g2 = random_pygeo_graph(n_node + 10, node_feat_dim, n_edge + 10, edge_feat_dim, device='cpu')
    summary(g1, 'g1')
    loader = DataLoader([g1] * 16 + [g2] * 16, batch_size=8, shuffle=True, num_workers=0)
    for batch in loader:
        pred = model(batch)
        summary(pred, 'pred')
