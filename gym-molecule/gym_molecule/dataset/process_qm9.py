import sys, os
from rdkit import Chem
from rdkit.Chem import QED
import glob


def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        smiles = lines[-2].split('\t')[0]
        return smiles


def process_data():
    all_files = glob.glob(os.path.join('/Users/arkshi/Downloads/dsgdb9nsd.xyz', '*.xyz'))
    for id, file_path in enumerate(all_files):
        with open('qm9.smi', 'a') as f:
            str = read_xyz(file_path) + '\n'
            f.write(str)


if __name__ == "__main__":
    process_data()