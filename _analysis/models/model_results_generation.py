import sys
import os
import numpy as np
from rdkit import Chem
sys.path.append('../utils')
sys.path.append('../../_utils')
from plot_utils import *
from read_dataset import readStr_qm9, read_zinc, read_zinc1M
from smile_metrics import MolecularMetrics as mm
from utils import save_scores_priors, load_decoded_priors, calc_perc

folder = "generation/"

# take params
name = sys.argv[1]
file = sys.argv[2]
dataset = sys.argv[3]

if dataset == "zinc":
    trainingSet = read_zinc()
elif dataset == "zinc1M":
    trainingSet = read_zinc1M()
elif dataset == "qm9":
    trainingSet = readStr_qm9()
else:
    print("Error: dataset not recognized.")
trainingSet = trainingSet[5000:]


# make folder
try:
    os.makedirs(folder + name)
except OSError:
    print("Creation of the directory %s failed" % (folder + name))
else:
    print("Successfully created the directory %s " % (folder + name))

# READ SMILES
smi = dict()
smi['decoded'] = load_decoded_priors(file)
smi['valid'] = []
for line in smi['decoded']:
    list = []
    for l in line:
        if mm.valid_lambda(l):
            list.append(l)
    smi['valid'].append(list)

# MAKES FLAT VERSION
flat = dict()
flat['smi'] = []
for line in smi['valid']:
    for l in line:
        flat['smi'].append(l)

# GET VALUES
values = []
values.append([mm.valid_binary_score(smi['decoded'][index])
                       for index in range(len(smi['decoded']))])
values.append(mm.novel_binary_score(flat['smi'], trainingSet))
values.append(mm.unique_total_score(flat['smi']))
values.append(mm.diversity_scores(flat['smi'], trainingSet))

values.append(mm.natural_product_scores(flat['smi'], norm=True))
values.append(mm.water_octanol_partition_coefficient_scores(flat['smi'], norm=True))
values.append(mm.synthetic_accessibility_score_scores(flat['smi'], norm=True))
values.append(mm.quantitative_estimation_druglikeness_scores(flat['smi'], norm=True))

# GET MEAN
scores = []
scores.append(calc_perc(np.mean(values[0])))  # the matrix is fixed so the mean is the global mean
scores.append(calc_perc(np.mean(values[1])))
scores.append(calc_perc(np.mean(values[2])))
scores.append(calc_perc(np.mean(values[3])))
scores.append(calc_perc(np.mean(values[4])))
scores.append(calc_perc(np.mean(values[5])))
scores.append(calc_perc(np.mean(values[6])))
scores.append(calc_perc(np.mean(values[7])))

# GET VARIANCE
errors = []
errors.append(calc_perc(np.std(values[0])))  # the matrix is fixed so the variance is the global variance e.g as flat
errors.append(calc_perc(np.std(values[1])))
errors.append(float('nan'))
errors.append(calc_perc(np.std(values[3])))
errors.append(calc_perc(np.std(values[4])))
errors.append(calc_perc(np.std(values[5])))
errors.append(calc_perc(np.std(values[6])))
errors.append(calc_perc(np.std(values[7])))


# SAVE SCORE
save_scores_priors(scores, folder + name + '/scores.txt')
save_scores_priors(errors, folder + name + '/scores_errors.txt')

