# This .py code is from molgrpah.py of the paper "Convolutional Networks on Graphs
# for Learning Molecular Fingerprints"

degrees = [0, 1, 2, 3, 4, 5]
import numpy as np
class MolGraph(object):
    def __init__(self):
        self.nodes = {}  # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None, neighbour = None):
        new_node = Node(ntype, features, rdkit_ix)
        new_node.add_neighbors(neighbour)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i: [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']]) ##?

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]



class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'idx']
    def __init__(self, ntype, features, idx):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.idx = idx

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

    def get_all_neibors(self):
        return self._neighbors

class Mol(object):
    __slots__ = ['num_atoms', 'atom_list']
    def __init__(self,num_atoms,atom_list):
        self.num_atoms=num_atoms
        self.atom_list=atom_list
