__author__ = "Bowen Liu"
__copyright__ = "Copyright 2018, Stanford University"

import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdmolops
import numpy as np
import pickle
import sys
import os
# def mol_to_nx(mol):
#     G = nx.Graph()
#
#     for atom in mol.GetAtoms():
#         G.add_node(atom.GetIdx(),
#                    symbol=atom.GetSymbol(),
#                    atomic_num=atom.GetAtomicNum(),
#                    formal_charge=atom.GetFormalCharge(),
#                    chiral_tag=atom.GetChiralTag(),
#                    hybridization=atom.GetHybridization(),
#                    num_explicit_hs=atom.GetNumExplicitHs(),
#                    is_aromatic=atom.GetIsAromatic())
#     for bond in mol.GetBonds():
#         G.add_edge(bond.GetBeginAtomIdx(),
#                    bond.GetEndAtomIdx(),
#                    bond_type=bond.GetBondType())
#     return G
bond_dict = {'SINGLE':0, 'DOUBLE':1, 'TRIPLE':2, 'AROMATIC':3}


def get_atom_types(dataset):
    atom_types = []
    if dataset == 'gdb':
        atom_types = ['C', 'N', 'O', 'S', 'Cl']  # gdb 13
    elif dataset == 'zinc':
        atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl',
                      'Br']  # ZINC
    elif dataset == 'qm9':
        atom_types = ['H', 'C', 'N', 'O', 'F']  # qm9
    return atom_types


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol


def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def load_dataset(path):
  """
  Loads gdb13 dataset from path to pandas dataframe
  :param path:
  :return:
  """
  df = pd.read_csv(path, header=None, names=['smiles'])
  return df

def sort_dataset(in_path, out_path):
    """
    Sorts the dataset of smiles from input path by molecular complexity as
    proxied by the BertzCT index, and outputs the new sorted dataset
    :param in_path:
    :param out_path:
    :return:
    """
    def _calc_bertz_ct(smiles):
        return GraphDescriptors.BertzCT(Chem.MolFromSmiles(smiles))

    in_df = load_dataset(in_path)
    in_df['BertzCT'] = in_df.smiles.apply(_calc_bertz_ct)
    sorted_df = in_df.sort_values(by=['BertzCT'])
    sorted_df['smiles'].to_csv(out_path, index=False)


class gdb_dataset:
  """
  Simple object to contain the gdb13 dataset
  """
  def __init__(self, path):
    self.df = load_dataset(path)

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, item):
    """
    Returns an rdkit mol object
    :param item:
    :return:
    """
    smiles = self.df['smiles'][item]
    mol = Chem.MolFromSmiles(smiles)
    return mol


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z


def find_carbon_idx(dataset, nodes):
    has_carbon = False
    for i in range(len(nodes)):
        if dataset == 'qm9':
            if np.argmax(nodes[i]) == 1:
                has_carbon = True
                return i
        elif dataset == 'zinc':
            if np.argmax(nodes[i]) == 0:
                has_carbon = True
                return i
    if not has_carbon:
        print("no carbon!")
        return 0


def to_graph(mol, dataset):
    if mol is None:
        return [], []
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None
    Chem.RemoveStereochemistry(mol)
    edges = []
    nodes = []
    atom_types = get_atom_types(dataset)
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
        assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        nodes.append(onehot(atom_types.index(str(atom.GetSymbol())), len(atom_types)))
    return nodes, edges


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj
