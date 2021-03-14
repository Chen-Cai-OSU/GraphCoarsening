# Created at 2020-05-26
# Summary: visualization

import argparse
import sys

import graph_coarsening
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import pygsp
import torch
from graph_coarsening.coarsening_utils import *
from scipy.sparse import lil_matrix
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx

from sparsenet.util.loss_util import vec_generator
from sparsenet.util.util import tonp, fix_seed, pyg2gsp, summary

gsp.plotting.BACKEND = 'matplotlib'


def plot3dpts(x):
    """ viz a numpy array of shape (n, 3)
        works for matplotlib==3.1.3
        For newer version, AttributeError: module 'matplotlib' has no attribute 'projections'

    """

    assert x.shape[1] == 3
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.show()


def plot_example(face, pos):
    """
    :param face: array of shape (3, n_face) (face index)
    :param pos: array coordinates of shape (n_pts, 3)
    :return:
    """
    assert face.shape[0] == pos.shape[1] == 3
    fig = go.Figure(data=[
        go.Mesh3d(
            x=pos[:, 0].T,
            y=pos[:, 1].T,
            z=pos[:, 2].T,

            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=face[0, :],
            j=face[1, :],
            k=face[2, :],
            name='y',
            showscale=True
        )
    ])
    fig.show()


def viz_graph(g, node_size=5, edge_width=1, node_color='b', color_bar=False, show=False):

    if isinstance(g, Data):
        if g.pos is not None:
            print('use intrinsic position')
            pos = tonp(g.pos)
        else:
            pos = None
        g = to_networkx(g)
    else: # nx graph
        pos = None

    if pos is None:
        print('use spring layout for visualization')
        pos = nx.spring_layout(g, seed=42)

    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width=edge_width)
    if color_bar:
        # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm._A = []
        plt.colorbar(sm)
    if show: plt.show()


def matshow(x, var_name='x'):
    """ improved matshow for better height/width ratio """
    assert isinstance(x, np.ndarray)
    summary(x)
    plt.imshow(x)
    if x.shape[0] > x.shape[1] * 10:
        plt.axes().set_aspect('auto')  # http://bit.ly/37V4xL0
    plt.colorbar()
    plt.title(f'{var_name} {np.shape(x)}')
    plt.show()


@timefunc
def set_coord(gspG):
    if not hasattr(gspG, 'coords') :
        try:
            import networkx as nx
            graph = nx.from_scipy_sparse_matrix(gspG.W)
            pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
            gspG.set_coordinates(np.array(list(pos.values())))
        except ImportError:
            gspG.set_coordinates()
    return gspG


def set_Wsign(gspG, gspG2, test = True):
    n = gspG.N

    if test:
        W = np.random.random((n, n))
        W = (W + W.T) / 2 - 0.5
        summary(W, 'W', highlight=True)
        gspG.W_sign = lil_matrix(W)
    else:
        assert gspG2.N == gspG.N, f'Two gspG size differ. {gspG.N} vs. {gspG2.N}'
        W = gspG2.W - gspG.W
        gspG.W_sign = lil_matrix(W)

    return gspG


class visualizer:
    def __init__(self, g=None, args=None):
        self.g = g # origianl g
        self.gen = vec_generator()
        self.k = 40 if args is None else args.n_bottomk

    def compare(self, g1, g2):
        # fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        if isinstance(g1, Data):
            summary(g1, 'g1')  # g1 = from_networkx(g1)
        if isinstance(g2, Data):
            summary(g2, 'g2')  # g2 = from_networkx(g2)

        # plt.title('Original')
        plt.subplot(121)
        viz_graph(g1, show=False)

        plt.subplot(122)
        plt.title('After')
        viz_graph(g2, show=True)

    def cmp_eig(self, g, L2, L3, args, show=False, dir=None):
        """ compare the eigenvalues of sparsifed graph w/wo learning.
         :param L1 original graph. should be able to read precomputed eigs
         :param L2: compute eigs only once
         :param L3: compute eigs for different epochs
         """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        assert isinstance(g, Data)
        key = f'{str(args.lap)}_vals'
        if  key in g:
            v1 = tonp(g[key])[:self.k]
            self.v1 = v1
        else:
            raise Exception('Not able to get eigenvals from origianl graph.')

        v2 = self.gen.bottomk_vec(L2, None, self.k, which='SM', val=True).real
        v3 = self.gen.bottomk_vec(L3, None, self.k, which='SM', val=True).real

        ax.plot(v1, label=f'Eigvals of big graph')
        ax.plot(v2, label=f'Eigvals of small graph (no learning)')
        ax.plot(v3, label=f'Eigvals of small graph (learning)')
        ax.legend(loc='lower right')
        ax.set_title(f'Eigenvalues Comparasion.')
        if show: plt.show()
        if dir is not None: fig.savefig(dir, bbox_inches='tight')


    def test(self):
        g = nx.random_geometric_graph(1000, 0.1)
        viz_graph(g, show=True)

    def plot_weight_diff(self, Gc, title='', edge_width=0.8, alpha=0.55, node_size=1, show=False, dir=None):

        def sign_scale(_x):
            if _x > 0:
                return 1 # _x
            else:
                return 1 #3 * (-_x)

        fig = plt.figure()
        fig.tight_layout()
        isinstance(Gc, pygsp.graphs.graph.Graph)
        assert Gc.W_sign is not None

        Gc = set_coord(Gc)
        edges_c = np.array(Gc.get_edge_list()[0:2])

        if Gc.coords.shape[1] == 2:
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            [x, y] = Gc.coords.T
            ax.scatter(x, y, c="k", s=node_size, alpha=alpha)
            for eIdx in range(0, edges_c.shape[1]):
                diff = Gc.W_sign[tuple(edges_c[:, eIdx])]
                ax.plot(
                    x[edges_c[:, eIdx]],
                    y[edges_c[:, eIdx]],
                    color="b" if diff > 0 else 'r',
                    alpha=alpha,
                    lineWidth=edge_width * sign_scale(diff),
                )
        elif Gc.coords.shape[1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.axis("off")
            [x, y, z] = Gc.coords.T
            ax.scatter(x, y, z, c="k", s=node_size, alpha=alpha)
            for eIdx in range(0, edges_c.shape[1]):
                diff = Gc.W_sign[tuple(edges_c[:, eIdx])]
                ax.plot(
                    x[edges_c[:, eIdx]],
                    y[edges_c[:, eIdx]],
                    z[edges_c[:, eIdx]],
                    color="b" if diff > 0 else 'r',
                    alpha=alpha,
                    lineWidth=edge_width * sign_scale(diff),
                )
        ax.set_title(title)
        if show: plt.show()
        if dir is not None:
            fig.savefig(dir, bbox_inches='tight')

    def viz_weight_diff(self, g1, g2, title=None, show=False, dir=None):
        """
        visualize the weight difference of graph g1 and g2
        :param g1: pyg or pygsp graph
        :param g2: pyg or pygsp graph
        :return:
        """

        if isinstance(g1, Data):
            summary(g1, 'inside viz_weigh_diff g1.', highlight=True)
            if 'pos' in g1.keys:
                pos = g1.pos
                g1 = pyg2gsp(g1)
                g1.coords = tonp(pos)
            else:
                g1 = pyg2gsp(g1)
        if isinstance(g2, Data):
            summary(g2, 'inside viz_weigh_diff g2.', highlight=True)
            g2 = pyg2gsp(g2)

        g1.W_sign = g1.W - g2.W

        assert g1.W_sign.nnz <= g2.W.nnz, f'g1.W_sign has {g1.W_sign.nnz} nnz. g2.W has {g2.W.nnz} nnz.'
        self.plot_weight_diff(g1, title=title, show=show, dir=dir)


def visualize_graphpair(G, G_prime, Assignment, reproducible=True):
    """
    :param G: Original G in networkx format
    :param G_prime: Sampled graph Gprime.
    :param Assignment: The correspondence dict from sample_N2Nlandmarks function.
    :param reproducible: fix seed?
    :return:
    """
    if reproducible:
        fix_seed()

    # layout method get pos dict
    coordinates = nx.drawing.spring_layout(G)
    subcoordinates = {key: np.array([0, 0], dtype=float) for key in list(G_prime.nodes)}

    # get subgraph layout coords using assignment.
    inverse_assignment = {v: key for key, value in Assignment.items() for v in value}
    for v in list(G.nodes):
        vp = inverse_assignment[v]
        subcoordinates[vp] += coordinates[v]
    for key in subcoordinates:
        subcoordinates[key] /= len(Assignment[key])
    print(subcoordinates)

    # draw node and edges.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for e in G.edges:
        x, y = [coordinates[e[0]][0], coordinates[e[1]][0]], [coordinates[e[0]][1], coordinates[e[1]][1]]
        ax1.plot(
            x, y, color="k", lineWidth=2,
        )
    x, y = [c[0] for c in coordinates.values()], [c[1] for c in coordinates.values()]
    ax1.scatter(x, y, s=2)
    ax1.set_title('G')

    for e in G_prime.edges:
        x, y = [subcoordinates[e[0]][0], subcoordinates[e[1]][0]], [subcoordinates[e[0]][1], subcoordinates[e[1]][1]]
        ax2.plot(
            x, y, color="k", lineWidth=2,
        )
    x, y = [c[0] for c in subcoordinates.values()], [c[1] for c in subcoordinates.values()]
    ax2.scatter(x, y, s=2)
    ax2.set_title('Sampled_G')

    plt.show()


@timefunc
def plot_coarsening(
        Gall, Call, size=3, edge_width=0.8, node_size=20, alpha=0.55, title=""
):
    """
    Plot a (hierarchical) coarsening

    Parameters
    ----------
    G_all : list of pygsp Graphs
    Call  : list of np.arrays

    Returns
    -------
    fig : matplotlib figure
    """

    # colors signify the size of a coarsened subgraph ('k' is 1, 'g' is 2, 'b' is 3, and so on)
    colors = ["k", "g", "b", "r", "y"]

    n_levels = len(Gall) - 1
    if n_levels == 0:
        return None
    fig = plt.figure(figsize=(n_levels * size * 3, size * 2))

    for level in range(n_levels):

        G = Gall[level]
        edges = np.array(G.get_edge_list()[0:2])

        Gc = Gall[level + 1]
        edges_c = np.array(Gc.get_edge_list()[0:2])
        C = Call[level]
        C = C.toarray()

        if G.coords.shape[1] == 2:
            ax = fig.add_subplot(1, n_levels + 1, level + 1)
            ax.axis("off")
            ax.set_title(f"{title} | level = {level}, N = {G.N}")

            [x, y] = G.coords.T

            # plot edges
            for eIdx in range(0, edges.shape[1]):
                color = "k"
                try:
                    color = "k" if G.W_sign[tuple(edges[:, eIdx])] > 0 else 'r'
                except AttributeError:
                    summary('', 'no G.W_sign', highlight=True)
                    pass

                ax.plot(
                    x[edges[:, eIdx]],
                    y[edges[:, eIdx]],
                    color=color,  # "k" if G.W[tuple(edges[:, eIdx])] > 0 else 'r',
                    alpha=alpha,
                    lineWidth=edge_width,
                )

            # plot nodes
            for i in range(Gc.N):
                subgraph = np.arange(G.N)[C[i, :] > 0]
                ax.scatter(
                    x[subgraph],
                    y[subgraph],
                    c=colors[np.clip(len(subgraph) - 1, 0, 4)],
                    s=node_size * len(subgraph),
                    alpha=alpha,
                )
        else:
            raise NotImplementedError

    # the final graph
    Gc = Gall[-1]
    edges_c = np.array(Gc.get_edge_list()[0:2])

    if G.coords.shape[1] == 2:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1)
        ax.axis("off")
        [x, y] = Gc.coords.T
        ax.scatter(x, y, c="k", s=node_size, alpha=alpha)
        for eIdx in range(0, edges_c.shape[1]):
            ax.plot(
                x[edges_c[:, eIdx]],
                y[edges_c[:, eIdx]],
                color="k",
                alpha=alpha,
                lineWidth=edge_width,
            )
    else:
        raise NotImplementedError

    ax.set_title(f"{title} | level = {n_levels}, n = {Gc.N}")
    fig.tight_layout()
    return fig


parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
# model
parser.add_argument('--dataset', type=str, default='random_geo', help='the name of dataset')

# optim
parser.add_argument('--n_bottomk', type=int, default=40, help='Number of Bottom K eigenvector')
parser.add_argument('--ratio', type=float, default=0.5, help='reduction ratio')
# debug
parser.add_argument('--lap', type=str, default='none', help='Laplacian type', choices=[None, 'sym', 'rw', 'none'])
parser.add_argument('--debug', action='store_true', help='debug. Smaller graph')
parser.add_argument('--tbx', action='store_true', help='write to tensorboardx.')
parser.add_argument('--viz', action='store_true', help='visualization of weights of sparsified graph ')
# parser.add_argument('--verbose', action='store_true', help='control the info level for real graph')
parser.add_argument('--train_idx', type=int, default=0, help='train index of the shape data. Do not change.')
parser.add_argument('--test_idx', type=int, default=0, help='test index of the shape data. Do not change.')
parser.add_argument('--lap_check', action='store_true', help='check the laplacian is normal during training')
parser.add_argument('--n_cycle', type=int, default=5, help='number of cycles')

parser.add_argument('--no_correction', action='store_true', help='Do not apply Laplacian correction')
parser.add_argument('--loukas_quality', action='store_true', help='Compute the coarsening quality of loukas method')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--train_indices', type=str, default='0,',
                    help='train indices of the dataset')  # https://bit.ly/3dtJtPn
parser.add_argument('--test_indices', type=str, default='0,', help='test indices of the dataset')
parser.add_argument('--strategy', type=str, default='DK', help='coarsening strategy', choices=['DK', 'loukas'])
parser.add_argument('--method', type=str, default='variation_edges', help='Loukas methods',
                    choices=['variation_neighborhoods', 'variation_edges', 'variation_cliques',
                             'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'variation_neighborhood'])


def plot_gsp(ax, gspG=None, edge_width=0.8, alpha=0.55, node_size=2, title=''):
    if gspG is None:
        gspG = graph_coarsening.graph_lib.real(500, 'yeast')

    gspG = set_coord(gspG)
    [x, y] = gspG.coords.T
    ax.scatter(x, y, c="k", s=node_size, alpha=alpha)

    edges_c = np.array(gspG.get_edge_list()[0:2])
    for eIdx in range(0, edges_c.shape[1]):
        ax.plot(
            x[edges_c[:, eIdx]],
            y[edges_c[:, eIdx]],
            color="b",
            alpha=alpha,
            lineWidth=edge_width,
        )
    ax.set_title(title)


if __name__ == '__main__':
    from sparsenet.util.data import data_loader

    args = parser.parse_args()
    pyg, _ = data_loader(args, dataset=args.dataset).load(args, mode='train')

    print(pyg.edge_index.size(1))
    pyg.edge_weight = torch.ones(pyg.edge_index.size(1))
    gspG = pyg2gsp(pyg)

    N = 600
    gspG = graph_coarsening.graph_lib.real(N, 'yeast')
    gspG = set_coord(gspG)
    summary(gspG.coords, 'gspG.coords', highlight=True)

    kwargs = {'K': 40, 'r': 0.5, 'method': 'heavy_edge', 'max_levels': 2}
    C, Gc, Call, Gall = coarsen(gspG, **kwargs)

    # Gall = [set_Wsign(gspG) for gspG in Gall]
    # visualizer().plot_weight_diff(Gall[-1], show=True, node_size=1)
    # gspG_noise = gspG
    # gspG_noise.W +=
    visualizer().viz_weight_diff(gspG, gspG, title='Weight Difference', show=True)
    exit()

    for n in [1000]:
        sizes = [int(n * 0.5), int(n * 0.5)]
        probs = [[5/n,  0.3/n],
                 [0.3/n, 5/n]]
        g = nx.stochastic_block_model(sizes, probs, seed=42)
        subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
        g = subgraphs[0]
        assert nx.is_connected(g), f'g is not connected'
        viz_graph(g, show=True)
    sys.exit()

    g = nx.random_geometric_graph(512 * 2, 0.1 / np.sqrt(2))
    viz_graph(g, show=True)

    g = nx.random_geometric_graph(512 * 4, 0.1 / np.sqrt(4))
    viz_graph(g, show=True)

    exit()

    n = 4
    scale = 3
    fig = plt.figure(figsize=(scale * n, scale * n))
    for i in range(n ** 2):
        ax = fig.add_subplot(n, n, i + 1)
        ax.axis("off")
        plot_gsp(ax, gspG=None)
        # ax.scatter([1,2], [3,4], c="k")

    plt.show()

    exit()


    args = parser.parse_args()
    pyg, _ = data_loader(args, dataset=args.dataset).load(args, mode='train')

    print(pyg.edge_index.size(1))
    pyg.edge_weight = torch.ones(pyg.edge_index.size(1))
    gspG = pyg2gsp(pyg)

    N = 600
    gspG = graph_coarsening.graph_lib.real(N, 'yeast')
    gspG = set_coord(gspG)
    summary(gspG.coords, 'gspG.coords', highlight=True)

    kwargs = {'K': 40, 'r': 0.5, 'method': 'heavy_edge', 'max_levels': 2}
    C, Gc, Call, Gall = coarsen(gspG, **kwargs)

    Gall = [set_Wsign(gspG) for gspG in Gall]
    visualizer().plot_weight_diff(Gall[-1])
    exit()
    plot_coarsening(Gall, Call, size=2, edge_width=0.8, node_size=20, alpha=0.55, title=f"heavy_edge")

    exit()

    methods = ['heavy_edge', 'variation_edges', 'variation_neighborhoods', 'algebraic_JC', 'affinity_GS', 'kron']
    for method in methods[:]:
        kwargs = {'K': 40, 'r': 0.5, 'method': method, 'max_levels': 20}
        C, Gc, Call, Gall = coarsen(gspG, **kwargs)
        plot_coarsening(Gall, Call, size=2, edge_width=0.8, node_size=20, alpha=0.55, title=f"{method}")
        plt.show()
    sys.exit()

    gprime, assignment = sample_N2Nlandmarks(g, N=5)
    visualize_graphpair(g, gprime, assignment)

    exit()
    g1, g2 = nx.random_geometric_graph(100, 0.1, seed=1), nx.random_geometric_graph(200, 0.1, seed=2)
    visualizer().compare(g1, g2)
    exit()

    import numpy as np

    x = np.random.randint(low=1, high=100, size=(100, 3))
    plot3dpts(x)
    sys.exit()

    pos = np.array([[0.1, 1, 2, 0], [0, 0, 1, 2], [0, 2, 0, 1]]).T
    face = np.array([[0, 0, 0, 1], [1, 2, 3, 2], [2, 3, 1, 3]])
    print(pos.shape, face.shape)
    # print(face)
    # print(face[0,:])
    plot_example(pos=pos, face=face)
    sys.exit()

    # Download data_ set from plotly repo
    pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
    # x, y, z = pts.T
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
    z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
    fig.show()

    sys.exit()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
    z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    sys.exit()
