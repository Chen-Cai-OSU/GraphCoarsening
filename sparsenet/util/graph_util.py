# Created at 2020-04-16
# Summary: util functions

import shutil
from warnings import warn

import numpy as np
import torch
import torch_geometric
from memory_profiler import profile
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import from_networkx, subgraph

from sparsenet.model.loss import get_laplacian_mat, get_sparse_C
from sparsenet.util.gsp_util import gsp2pyg
from sparsenet.util.name_util import set_coarsening_graph_dir
from sparsenet.util.sample import sample_N2Nlandmarks
from sparsenet.util.util import timefunc, banner, summary, random_edge_index, fix_seed, red, make_dir

INFINITY = 1e8


@timefunc
def get_bipartite(G1, G2, crossing_edge):
    '''
    :param G1: graph 1
    :param G2: graph 2
    :param crossing_edge: crossing edges between those two subgraphs of G
    :return: A bipartiti graph G1 <-> G2. (nodes(G1) + nodes(G2) + crossing edge) in format of torch_geo
    '''
    xedge_index, xedge_attr = crossing_edge
    final_x = torch.cat((G1.x, G2.x), 0)
    final_node_index = torch.cat((G1.node_index, G2.node_index), 0)
    return Data(x=final_x, edge_attr=xedge_attr, edge_index=xedge_index, node_index=final_node_index)


@timefunc
def get_merged_subgraph(G1, G2, crossing_edge):
    '''
    :param G1: pygeo graph G1
    :param G2: pygeo graph G2
    :param crossing_edge: (edge_index, edge_attr)
    :return: Merge x, edge_attr, edge_index, and node_index in G1, G2, and crossing edge
    '''
    xedge_index, xedge_attr = crossing_edge
    final_edge_index = torch.cat((G1.edge_index, G2.edge_index, xedge_index), 1)
    final_edge_attr = torch.cat((G1.edge_attr, G2.edge_attr, xedge_attr), 0)
    final_x = torch.cat((G1.x, G2.x), 0)
    final_node_index = torch.cat((G1.node_index, G2.node_index), 0)
    return Data(x=final_x, edge_attr=final_edge_attr, edge_index=final_edge_index, node_index=final_node_index)


@profile
class GraphPair(object):
    def __init__(self, G, g_sml, Assignment):
        """
        :param G: Original graph G, in pygeo format
        :param gsml: sampled graph g_sml, in pygeo format, the node index of both G and gsml MUST starts with 0.
        :param Assignment: assignment
        """
        assert (isinstance(G, torch_geometric.data.data.Data)
                and isinstance(g_sml, torch_geometric.data.data.Data)), f'G is {type(G)}. g_sml is {type(g_sml)}'
        self.G = G
        if 'edge_attr' not in G.keys:
            # todo: discuss with DK
            warn(f'edge_attr does not exit. Will creat 1-dim edge attr (all ones)')
            G.edge_attr = torch.ones((G.num_edges, 1))
            banner('modified G')
            summary(G)

        self.G_prime = g_sml
        self.assignment = Assignment
        assert (self.__check_indexes()), 'Input graphs must have node_index starting from 0.'
        tensor_imap, inverse_assignment = {}, {}
        for i, (uprime, vprime) in enumerate(g_sml.edge_index.t().tolist()):
            if uprime > vprime:
                tensor_imap[(vprime, uprime)] = [i] if (vprime, uprime) not in tensor_imap.keys() \
                    else tensor_imap[(vprime, uprime)] + [i]
            else:
                tensor_imap[(uprime, vprime)] = [i] if (uprime, vprime) not in tensor_imap.keys() \
                    else tensor_imap[(uprime, vprime)] + [i]
        for key, value in Assignment.items():
            for v in value:
                inverse_assignment[v] = key
        self.tensor_imap = tensor_imap
        self.inverse_assignment = inverse_assignment  # map index of big graph to small graph

    def __check_indexes(self):
        gflag, gprimeflag = False, False
        for (u, v) in enumerate(self.G.edge_index.t().tolist()):
            if u == 0 or v == 0:
                gflag = True
                break
        for (uprime, vprime) in enumerate(self.G_prime.edge_index.t().tolist()):
            if uprime == 0 or vprime == 0:
                gprimeflag = True
                break
        return gflag and gprimeflag

    @timefunc
    def construct(self):
        """
        This function basically compute the partition of original graph G based on landmarks. Also, it will
        compute the crossing_edges, and the indices in tensor of any edge (u, v) in G_prime.
        :return: void.
        """
        N = len(self.assignment.keys())
        G_edge_attrs, G_edge_indices = self.G.edge_attr.tolist(), self.G.edge_index.t().tolist()
        G_x = self.G.x.tolist()
        print('N:', N, 'x:', len(G_x), 'edge_attr:', len(G_edge_attrs), 'edge_indices:', len(G_edge_indices))
        # subgraphs: list of subgraph: {'edge_index':[], 'edge_attr':[], 'x':[]}
        # crossing_edges: dictionary key (uprime, vprime) an edge from G_prime; crossing_edges[(up, vp)] = list of
        # [e=(u, v), e_attr], where e is an edge in G, and e_attr is its corresponding attr.
        subgraphs, crossing_edges = [{'edge_index': [], 'edge_attr': [], 'x': []} for i in range(N)], {}
        for i, (u, v) in enumerate(G_edge_indices):
            uprime, vprime = self.inverse_assignment[u], self.inverse_assignment[v]
            if uprime == vprime:  # add into subgraph gs[uprime]
                subgraphs[uprime]['edge_index'].append((u, v))
                subgraphs[uprime]['edge_attr'].append(G_edge_attrs[i])
            else:  # add into crossing edges [(up, vp)]
                uprime, vprime = (vprime, uprime) if uprime > vprime else (uprime, vprime)
                crossing_edges[(uprime, vprime)] = [[(u, v), G_edge_attrs[i]]] if (uprime, vprime) not in crossing_edges \
                    else crossing_edges[(uprime, vprime)] + [[(u, v), G_edge_attrs[i]]]

        for i, nx in enumerate(G_x):
            xprime = self.inverse_assignment[i]
            subgraphs[xprime]['x'].append(i)
        self.subgraphs = subgraphs
        self.crossing_edges = crossing_edges

    def __get_subgraph(self, uprime):
        """
        :param uprime: the id of a landmark in G_prime
        :return: the corresponding subgraph in original graph G (in pygeo format).
        """
        _x = torch.FloatTensor(self.subgraphs[uprime]['x'])
        _edge_index = torch.LongTensor(self.subgraphs[uprime]['edge_index']).t()
        _edge_attr = torch.FloatTensor(self.subgraphs[uprime]['edge_attr'])
        _node_index = torch.LongTensor(list(self.assignment[uprime]))
        return Data(x=_x, edge_index=_edge_index, edge_attr=_edge_attr, node_index=_node_index)

    def __get_crossing_edges(self, uprime, vprime):
        """
        :param uprime: landmark uprime in G_prime
        :param vprime: landmark vprime in G_prime
        :return: the crossing edges between subgraphs G_u, G_v (assigned to u_prime & v_prime) in G,
        return None if (uprime, vprime) is not in self.crossing_edges
        """
        assert (uprime < vprime) and (uprime, vprime) in self.crossing_edges, f'({uprime}, {vprime}) is not in crossing' \
            f'edges dictionary.'
        data = self.crossing_edges[(uprime, vprime)]
        _edge_index = [e[0] for e in data]
        _edge_attr = [e[1] for e in data]
        return (torch.LongTensor(_edge_index).t(), torch.FloatTensor(_edge_attr))

    def __get_tensor_indices(self, uprime, vprime):
        """
        :param uprime: landmark uprime in G_prime
        :param vprime: landmark vprime in G_prime
        :return: the indices of edge (uprime, vprime) in tensor edge_index in g_sml.
        """
        assert (uprime < vprime and (uprime, vprime) in self.tensor_imap)
        return tuple(self.tensor_imap[(uprime, vprime)])

    def get_data(self, edge):
        """
        :param edge: (u, v) from the sampled graph.
        :return: subgraphs G1, G2 corresponding to landmark u, v in edge. Crossing edges between G1, G2
        and the indices of (u, v), (v, u) in tensor edge_index of g_sml.
        """
        uprime, vprime = edge
        uprime, vprime = (vprime, uprime) if uprime > vprime else (uprime, vprime)
        G1, G2 = self.__get_subgraph(uprime), self.__get_subgraph(vprime)
        if G1.num_nodes == 1:
            warn(f'edge {edge}: output subgraph G1 is a singleton!')
        if G2.num_nodes == 1:
            warn(f'edge {edge}: output subgraph G2 is a singleton!')
        crossing_edges = self.__get_crossing_edges(uprime, vprime)
        tensor_indices = self.__get_tensor_indices(uprime, vprime)
        return G1, G2, crossing_edges, tensor_indices


@profile
class subgraphs(object):
    def __init__(self, g, assertion=False, args=None):
        """
        :param g: pyG graph with edeg_weight
        :param assertion: assert the edge index is right.
        """

        self.g = g
        self.C = None
        self.args = args
        assert 'edge_weight' in g.keys
        assert isinstance(g, Data)

        self.__load_coarsening_graphs(args, recompute=False)

        self.graph_pair = GraphPair(self.g, self.g_sml, self.assignment)
        self.graph_pair.construct()
        dict = self.__construct_dict()
        del self.graph_pair

        self.edges, self.inv_edges, self.double_edges = [], [], []
        for (idx1, idx2) in dict.keys():
            self.edges.append(idx1)
            self.inv_edges.append(idx2)
            self.double_edges.append((idx1, idx2))  # [idx1, idx2]
            if assertion: self.__assert(idx1, idx2)

        new_info = {}
        for k, v in dict.items():
            # k is of form [8, 9], v is of form (G1, G2, (crossing_edge_index, crossing_edge_attr), ini)
            new_info[k[0]] = v
        self.info = new_info
        del new_info

    def __assert(self, idx1, idx2):
        set_e1 = set(self.g_sml.edge_index[:, idx1].numpy())  # {0,, 13}
        set_e2 = set(self.g_sml.edge_index[:, idx2].numpy())  # {13, 0}
        assert set_e1 == set_e2, f'{idx1} edge is {set_e1}. {idx2} edge is {set_e2}'

    def __load_coarsening_graphs(self, args, recompute=False):
        dir = set_coarsening_graph_dir(args)
        if recompute:
            shutil.rmtree(dir)
            make_dir(dir)

        if args.strategy == 'DK':
            n_sml = int(self.g.num_nodes * (1 - args.ratio))
            try:
                self.assignment = torch.load(f'{dir}assignment.pt')
                self.g_sml = torch.load(f'{dir}g_sml.pt')
                print(f'load g_sml, assignment from \n {red(dir)}')
            except FileNotFoundError:
                g_sml, assignment = sample_N2Nlandmarks(self.g, n_sml, weight_key='edge_weight')
                self.g_sml = from_networkx(g_sml)
                self.assignment = assignment

                # save g_sml, assignment
                torch.save(self.assignment, f'{dir}assignment.pt')
                torch.save(self.g_sml, f'{dir}g_sml.pt')
                print(f'save at g_sml, assignment at \n {red(dir)}')

            # todo: add a function to convert self.assignment to C
            self.C = get_sparse_C(self.g.num_nodes, n_sml, self.assignment)
        elif args.strategy == 'loukas':
            try:
                self.assignment = torch.load(f'{dir}assignment.pt')
                self.g_sml = torch.load(f'{dir}g_sml.pt')
                self.C = torch.load(f'{dir}C.pt')
                print(f'load g_sml, assignment, and C from \n {red(dir)}')
            except (FileNotFoundError, TypeError):
                loukas_kwargs = {'r': args.ratio, 'method': args.method,
                                 'loukas_quality': args.loukas_quality,
                                 'K': args.n_bottomk}
                converter = gsp2pyg(self.g, **loukas_kwargs)
                g_sml, assignment = converter.pyg_sml, converter.assignment
                self.g_sml = g_sml
                self.C = converter.C
                self.assignment = assignment

                # save g_sml, C, assignment
                torch.save(self.assignment, f'{dir}assignment.pt')
                torch.save(self.g_sml, f'{dir}g_sml.pt')
                torch.save(self.C, f'{dir}C.pt')
                print(f'save at g_sml, assignment, and C at \n {red(dir)}')
        else:
            raise NotImplementedError

    @timefunc
    def __construct_dict(self):
        """ construct a dict.
            modified from get_original_subgraphs where __map_back_from_edges is replaced by graph_pair.get_data,
            which makes it faster

        :return: a set of {edge_indexes (i, j):(G1, G2, (crossing_edge_index, crossing_edge_attr(DELETED)))
         """
        assert (isinstance(self.g, torch_geometric.data.data.Data) and isinstance(self.g_sml,
                                                                                  torch_geometric.data.data.Data))
        imap, ret = {}, {}
        for i, (u, v) in enumerate(self.g_sml.edge_index.t().tolist()):
            if u > v:
                # map node index (u, v) to edge index (i, j)
                imap[(v, u)] = [i] if (v, u) not in imap.keys() else imap[(v, u)] + [i]
            else:
                imap[(u, v)] = [i] if (u, v) not in imap.keys() else imap[(u, v)] + [i]
        for _, (u, v) in enumerate(self.g_sml.edge_index.t().tolist()):
            if u < v:
                (i, j) = tuple(imap[(u, v)])
                G1, G2, crossing_edges, _ = self.graph_pair.get_data(
                    (u, v))  # __map_back_from_edge(G, (u, v), assignment)
                ini = self.g_sml.edge_weight[[i, j]]  # used to initialize gnn output. assert ini[0]==ini[1]
                ret[(i, j)] = (G1, G2, crossing_edges, ini)
        return ret

    @timefunc
    def get_subgraphs(self, verbose=False):
        """ the main fucntions that is called
            return a list of pyG graph corresponding to each edge in G'
        """

        subgraphs_list = []

        for e in self.edges:
            G1, G2, crossing_edges, ini = self.info[e]
            pyG = get_merged_subgraph(G1, G2, crossing_edges)
            indices = pyG.node_index.numpy().ravel().tolist()

            try:
                new_edge_index, new_edge_attr = subgraph(indices, pyG.edge_index, pyG.edge_attr, relabel_nodes=True)
            except IndexError:
                warn('Index Error. Filter out isolated notes.')
                _edge_indices = pyG.edge_index.numpy().ravel().tolist()
                indices = [idx for idx in indices if idx in _edge_indices]
                new_edge_index, new_edge_attr = subgraph(indices, pyG.edge_index, pyG.edge_attr, relabel_nodes=True)

            new_pyG = Data(edge_index=new_edge_index, edge_attr=new_edge_attr, ini=torch.ones(1) * ini[0])
            new_pyG.x = None
            new_pyG = LocalDegreeProfile()(new_pyG)
            new_pyG.x = Variable(new_pyG.x)
            new_pyG.x = torch.nn.functional.normalize(new_pyG.x, dim=0)
            subgraphs_list += [new_pyG]

        del self.info
        if verbose:
            for idx, v in enumerate(subgraphs_list):
                summary(v, f'{idx}-subgraph')

        print(f'{len(subgraphs_list)} Subgraph Stats:')
        nodes_stats = [g_.num_nodes for g_ in subgraphs_list]
        edges_stats = [g_.num_edges for g_ in subgraphs_list]
        summary(np.array(nodes_stats), 'node_stats')
        summary(np.array(edges_stats), 'edge_stats')

        return subgraphs_list

    def get_bipartitle_graphs(self):
        # todo: similar to get_subgraphs but for bipartitle graph. 0.
        raise NotImplementedError

    def baseline0(self, normalization):
        """
        return the laplacian of baseline 0, which is the Laplacian of G' without learning
        Summary of g_sml in baseline0 (torch_geometric.data.data.Data):
               edge_index               LongTensor          [2, 9476]      796.55(mean)   0.0(min) 1688.0(max) 770.0(median) 492.89(std) 1689.0(unique)
               edge_weight              FloatTensor         [9476]          2.11(mean)   1.0(min)  10.0(max)   2.0(median)  1.12(std)  10.0(unique)
        """
        g_sml = self.g_sml  # all index should be contiguous
        L = get_laplacian_mat(g_sml.edge_index, g_sml.edge_weight, g_sml.num_nodes, normalization=normalization)
        return L

    def L(self, g, normalization):
        # todo: add check num_nodes
        L = get_laplacian_mat(g.edge_index, g.edge_weight, g.num_nodes, normalization=normalization)
        return L

    def trivial_L(self, g):
        """ trival Laplacian for standarded Laplacian"""
        L = get_laplacian_mat(g.edge_index, torch.zeros(g.edge_weight.size()), g.num_nodes, normalization=None)
        return L


if __name__ == '__main__':
    # edge = random_edge_index(n_edge=200, n_node=20)
    fix_seed()

    n_node, n_edge = 3200, 40000
    node_dim = 1
    edge_feat_dim = 1
    n_node_sml = 200

    g = Data(x=torch.rand(n_node, node_dim),
             edge_index=random_edge_index(n_edge, n_node),
             edge_attr=torch.rand(n_edge, edge_feat_dim))
    g.edge_weight = torch.ones(n_edge) * 1.1
    summary(g, 'original_graph')

    # n_sml = 200
    # banner('Test subgraphs')
    # all_graphs = subgraphs(g, n_sml).get_subgraphs(verbose=False)

    banner('Test sample Landmark')
    g_sml, assignment = sample_N2Nlandmarks(g, n_node_sml, weight_key='edge_weight')
    print(g_sml.edges.data())
    g_sml_pyG = from_networkx(g_sml)

    banner('Get original subgraphs test')
    graph_pair = GraphPair(g, g_sml_pyG, assignment)
    graph_pair.construct()

    edge_indexes = g_sml_pyG.edge_index.t().tolist()

    for i, edge in enumerate(edge_indexes):
        print(edge)
        G1, G2, crossing_edge, tensor_indices = graph_pair.get_data(edge)
        if i > 4: exit()
        continue

        # When G1 or G2 is a single node, summary() will cause error.
        # summary(G1, 'G1')
        print(G2)
        # exit()
        # summary(G2, 'G2')
        print(crossing_edge[0].shape, crossing_edge[1].shape)
        print(tensor_indices)

    # edge_indexes = g_sml_pyG.edge_index.numpy()
    # dict = get_original_subgraphs(g, g_sml_pyG, assignment)
    # for i, j in dict.keys():
    #     print('edge_index_pair ({}, {}):'.format(i, j), edge_indexes[:, i], edge_indexes[:, j])
    #
    # banner('g_sml_nx')
    # summary(g_sml_pyG, 'g_sml_pyG')
    # edges = [e for e in g_sml.edges]
    # summary(torch.Tensor(np.array(edges)), 'g_sml_nx')
    #
    # banner('DK\'s test')
    # print('Select edge:', edges[0])
    # G1, G2, crossing_edges = __map_back_from_edge(g, edges[0], assignment)
    # summary(G1, 'G1')
    # summary(G2, 'G2')
    # print('Crossing edge size:', crossing_edges[0].shape)
    #
    # summary(get_bipartite(G1, G2, crossing_edges), 'bipartitle')
    # summary(get_merged_subgraph(G1, G2, crossing_edges), 'merged_subgraph')
