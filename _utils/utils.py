from read_dataset import read_qm9Original, read_qm9, readStr_qm9, read_zinc
from rdkit import Chem
import numpy as np


def save_decoded_results(smiles, results, filename):
    f = open(filename, "w")
    for smile, line in zip(smiles, results):
        f.write(smile)
        f.write(";,;")
        for res in line:
            f.write(res)
            f.write(";,;")
        f.write("\n")
    f.close()

def save_decoded_priors(results, filename):
    f = open(filename, "w")
    for line in results:
        for res in line:
            f.write(res)
            f.write(";,;")
        f.write("\n")
    f.close()

def load_decoded_results(file):
    f = open(file, "r")
    decoded_res = []
    XTE = []
    for l in f:
        if l != "":
            lsani = l.strip()
            lsani = lsani.strip(";,;")
            smiles = lsani.split(";,;")
            XTE.append(smiles[0])
            decoded_res.append(smiles[1:])
    return XTE, np.array(decoded_res)

def load_decoded_priors(file):
    f = open(file, "r")
    decoded_res = []
    for l in f:
        if l != "":
            lsani = l.strip()
            lsani = lsani.strip(";,;")
            smiles = lsani.split(";,;")
            decoded_res.append(smiles)
    return np.array(decoded_res)


def save_scores_priors(results, file):
    f = open(file, "w")
    f.write("Validity score: {}\n".format(results[0]))
    f.write("Novelty score mean: {}\n".format(results[1]))
    f.write("Uniqueness score mean: {}\n".format(results[2]))
    f.write("Diversity score: {}\n".format(results[3]))
    f.write("Natural product score: {}\n".format(results[4]))
    f.write("Water octanol partition coefficient score: {}\n".format(results[5]))
    f.write("Synthetic accessibility score: {}\n".format(results[6]))
    f.write("Quantitative_estimation_druglikeness score: {}\n".format(results[7]))
    f.close()

def save_scores_bias(results, file):
    f = open(file, "w")
    f.write("Reconstruction score mean: {}\n".format(results[0]))
    f.write("Validity score: {}\n".format(results[1]))
    f.write("Novelty score mean: {}\n".format(results[2]))
    f.write("Uniqueness score mean: {}\n".format(results[3]))
    f.write("Diversity score: {}\n".format(results[4]))
    f.write("Natural product score: {}\n".format(results[5]))
    f.write("Water octanol partition coefficient score: {}\n".format(results[6]))
    f.write("Synthetic accessibility score: {}\n".format(results[7]))
    f.write("Quantitative_estimation_druglikeness score: {}\n".format(results[8]))
    f.close()

def zincProve():
    L = read_zinc()
    MAX = 120
    count = 0
    countDot = 0
    countAst = 0
    nMolMIn9 = 0
    nMaxAtoms = 0
    nMinAtom= 100

    nMol1 = 0

    for s in L:
        if len(s) > MAX:
            count = count + 1
        if "." in s:
            countDot = countDot + 1
        if "*" in s:
            countAst = countAst + 1
        m = Chem.MolFromSmiles(s)
        atom = m.GetNumAtoms()
        if atom <= 9:
            nMolMIn9 = nMolMIn9 + 1
        if atom > nMaxAtoms:
            nMaxAtoms = atom
        elif atom < nMinAtom:
            nMinAtom = atom

        if atom == 1:
            nMol1 = nMol1 + 1

    print("Numero molecole con num. atomi <= 9: {}".format(nMolMIn9))
    print("Numero massimo di molecole: {}".format(nMaxAtoms))
    print("Numero minimo di molecole: {}".format(nMinAtom))
    print("Numero molecole con caratteri superiori a 120: {}".format(count))
    print("Numero molecole con carattere '.' : {}".format(countDot))
    print("Numero molecole con carattere '*' : {}".format(countAst))
    print("Numero di molecole lette: {}".format(len(L)))
    print("Numero di molecole lette formate da un solo atomo: {}".format(nMol1))
    print("-------------- FINE ---------------")

def qm9OriginalProve():
    L = read_qm9Original()
    nMolMIn9 = 0
    nMaxAtoms = 0
    nMinAtom= 100

    nMol1 = 0

    for m in L:
        atom = m.GetNumAtoms()
        if atom <= 9:
            nMolMIn9 = nMolMIn9 + 1
        if atom > nMaxAtoms:
            nMaxAtoms = atom
        elif atom < nMinAtom:
            nMinAtom = atom

        if atom == 1:
            nMol1 = nMol1 + 1

    print("Numero molecole con num. atomi <= 9: {}".format(nMolMIn9))
    print("Numero massimo di atomi: {}".format(nMaxAtoms))
    print("Numero minimo di atomi: {}".format(nMinAtom))
    print("Numero di molecole lette: {}".format(len(L)))
    print("Numero di molecole lette formate da un solo atomo: {}".format(nMol1))
    print("-------------- FINE ---------------")

def qm9Prove():
    L = read_qm9()
    MAX = 120
    count = 0
    countDot = 0
    countAst = 0
    nMolMIn9 = 0
    nMaxAtoms = 0
    nMinAtom= 100

    nMol1 = 0

    for s in L:
        if len(s) > MAX:
            count = count + 1
        if "." in s:
            countDot = countDot + 1
        if "*" in s:
            countAst = countAst + 1
        m = Chem.MolFromSmiles(s)
        atom = m.GetNumAtoms()
        if atom <= 9:
            nMolMIn9 = nMolMIn9 + 1
        if atom > nMaxAtoms:
            nMaxAtoms = atom
        elif atom < nMinAtom:
            nMinAtom = atom

        if atom == 1:
            nMol1 = nMol1 + 1

    print("Numero molecole con num. atomi <= 9: {}".format(nMolMIn9))
    print("Numero massimo di atomi: {}".format(nMaxAtoms))
    print("Numero minimo di atomi: {}".format(nMinAtom))
    print("Numero molecole con caratteri superiori a 120: {}".format(count))
    print("Numero molecole con carattere '.' : {}".format(countDot))
    print("Numero molecole con carattere '*' : {}".format(countAst))
    print("Numero di molecole lette: {}".format(len(L)))
    print("Numero di molecole lette formate da un solo atomo: {}".format(nMol1))
    print("-------------- FINE ---------------")

def qm9StrProve():
    L = readStr_qm9()
    MAX = 120
    count = 0
    countDot = 0
    countAst = 0
    nMolMIn9 = 0
    nMaxAtoms = 0
    nMinAtom= 100

    nMol1 = 0

    for s in L:
        if len(s) > MAX:
            count = count + 1
        if "." in s:
            countDot = countDot + 1
        if "*" in s:
            countAst = countAst + 1
        m = Chem.MolFromSmiles(s)
        atom = m.GetNumAtoms()
        if atom <= 9:
            nMolMIn9 = nMolMIn9 + 1
        if atom > nMaxAtoms:
            nMaxAtoms = atom
        elif atom < nMinAtom:
            nMinAtom = atom

        if atom == 1:
            nMol1 = nMol1 + 1

    print("Numero molecole con num. atomi <= 9: {}".format(nMolMIn9))
    print("Numero massimo di atomi: {}".format(nMaxAtoms))
    print("Numero minimo di atomi: {}".format(nMinAtom))
    print("Numero molecole con caratteri superiori a 120: {}".format(count))
    print("Numero molecole con carattere '.' : {}".format(countDot))
    print("Numero molecole con carattere '*' : {}".format(countAst))
    print("Numero di molecole lette: {}".format(len(L)))
    print("Numero di molecole lette formate da un solo atomo: {}".format(nMol1))
    print("-------------- FINE ---------------")



def translationQM9(file):
    from read_dataset import read_qm9

    D = read_qm9()
    # fix problem about molecule with '.' inside
    XTE = []
    for mol in D:
        if "." not in mol:
            XTE.append(mol)
    XTE = XTE[0:5000]

    f = open(file, "r")
    decoded_res = []
    for l in f:
        if l != "":
            lsani = l.strip()
            lsani = lsani.strip(";,;")
            smiles = lsani.split(";,;")
            decoded_res.append(smiles)

    save_decoded_results(XTE, decoded_res, file + "_new")


def translationZINC(file):
    from read_dataset import read_zinc

    XTE = read_zinc()
    XTE = XTE[0:5000]

    f = open(file, "r")
    decoded_res = []
    for l in f:
        if l != "":
            lsani = l.strip()
            lsani = lsani.strip(";,;")
            smiles = lsani.split(";,;")
            decoded_res.append(smiles)

    save_decoded_results(XTE, decoded_res, file + "_new")


def createSubSetQM9(n=5000):
    from read_dataset import readStr_qm9

    D = readStr_qm9()
    np.random.shuffle(D)
    subSet = D[:n]

    f = open("qm9_sub" + str(n) + ".smi", "w")
    for l in subSet:
        f.write(l)
        f.write("\n")
    f.close()


def createSubSetZINC(n=5000):
    from read_dataset import read_zinc

    D = read_zinc()
    np.random.shuffle(D)
    subSet = D[:n]

    f = open("250k_rndm_zinc_drugs_clean_sub" + str(n) + ".smi", "w")
    for l in subSet:
        f.write(l)
        f.write("\n")
    f.close()


def calc_perc(val, r=2):
    val = val * 100
    return round(val, r)


if __name__ == '__main__':
    createSubSetZINC()
    createSubSetQM9()

