from rdkit import Chem
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

def read_qm9Original():
    filename = current_dir + '/../_dataset/QM9/gdb9.sdf'

    dataset = list(filter(lambda x: x is not None, Chem.SDMolSupplier(filename)))
    return dataset


def read_qm9():
    filename = current_dir + '/../_dataset/QM9/gdb9.sdf'

    dataset = Chem.SDMolSupplier(filename)
    L = []
    for i in dataset:
        if i is None:
            continue
        smile = Chem.MolToSmiles(i)
        L.append(smile)
    return L

# This is a subset of read_qm9() results. It doesn't include smiles with "." or "*" in it
def readStr_qm9():
    f = open(current_dir + '/../_dataset/QM9/qm9.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    return L

def read_zinc():
    f = open(current_dir + '/../_dataset/ZINC/250k_rndm_zinc_drugs_clean.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    return L
