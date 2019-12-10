import pickle
import gzip
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen

import math
import numpy as np
import os
dir = os.path.dirname(os.path.realpath(__file__))

NP_model = pickle.load(gzip.open(dir + '/../_dataset/QM9/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(dir + '/../_dataset/QM9/SA_score.pkl.gz')) for j in range(1, len(i))}


class MolecularMetrics(object):
    

    # Remap the number to [0, 1]
    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    # Function used to check if x is a valid molecule
    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolFromSmiles(x) is not None and x.strip() != ''

    # Function used to check if x is a valid molecule and it doesn't include '*' or '.'
    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolFromSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    # ----------- RECONSTRUCTION BLOCK ---------------
    # Function used to check if all the elements in the decoded_smiles are equal to the target,
    # returns the mean
    @staticmethod
    def reconstruction_total_score(target, decoded_smiles):
        return np.array([x == target for x in decoded_smiles], dtype=np.float32).mean()

    @staticmethod
    def reconstruction_binary_score(target, decoded_smiles):
        return np.array([x == target for x in decoded_smiles], dtype=np.float32)

    # ----------- VALID BLOCK ---------------
    # Returns only the valid molecules without  '*' or '.' in a float array
    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32)

    # Returns only the valid molecules
    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    # Returns the mean score of the valid molecule
    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def valid_binary_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32)

    # ----------- NOVEL BLOCK ---------------
    @staticmethod
    def novel_scores(mols, dataset):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and x not in dataset, mols)), dtype=np.float128)

    @staticmethod
    def novel_filter(mols, dataset):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and x not in dataset, mols))

    @staticmethod
    def novel_total_score(mols, dataset):
        mean = 0
        valid_list = MolecularMetrics.valid_filter(mols)
        if len(valid_list) != 0:
            mean = MolecularMetrics.novel_scores(valid_list, dataset).mean()
        return mean

    @staticmethod
    def novel_binary_score(mols, dataset):
        valid_list = MolecularMetrics.valid_filter(mols)
        return MolecularMetrics.novel_scores(valid_list, dataset)

    # ----------- UNIQUE BLOCK ---------------
    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: x if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(v)
        return 0 if len(v) == 0 else len(s) / float(len(v))

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def natural_product_scores(mols, norm=False):
        # for smiles data and not rdkit.Chem obj
        molsObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(mols)]

        # calculating the score
        scores = [sum(NP_model.get(bit, 0) for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,2).GetNonzeroElements())
                  / float(mol.GetNumAtoms()) if mol is not None and  float(mol.GetNumAtoms()) != 0 else None for mol in molsObj]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    # Function used to calculate Druglikeliness
    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        # for smiles data and not rdkit.Chem obj
        molsObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(mols)]

        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            molsObj])))

    # Function used to calculate Solubility
    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        # for smiles data and not rdkit.Chem obj
        molsObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(mols)]

        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in molsObj]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores


    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    # Function used to calculate Synthetizability
    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        # for smiles data and not rdkit.Chem obj
        molsObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(mols)]
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in molsObj]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        # for smiles data and not rdkit.Chem obj
        dataObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(data)]
        molsObj = [Chem.MolFromSmiles(smiles) for smiles in MolecularMetrics.valid_filter(mols)]

        rand_mols = np.random.choice(dataObj, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, molsObj)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):
        score1 = MolecularMetrics.constant_bump(MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210, 0.945)
        score2 = MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
        valid_list = MolecularMetrics.valid_filter(mols)
        score3 = MolecularMetrics.novel_scores(valid_list, data) # only valid molecule
        score4 = 1 - MolecularMetrics.novel_scores(valid_list, data) # only valid molecule
        scoresSum =score1 + score2 + score3 + score4 * 0.3
        scores = scoresSum / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))
