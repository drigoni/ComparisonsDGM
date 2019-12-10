import molecule_vae
from rdkit import Chem
import numpy as np
import os
import sys
sys.path.append('%s/../_utils' % os.path.dirname(os.path.realpath(__file__)))
from smile_metrics import MolecularMetrics
from read_dataset import read_qm9
from utils import save_decoded_results, save_decoded_priors, load_decoded_results

# constant
GRAMMAR_WEIGHTS = "results/qm9_vae_grammar_L56_E100_val.hdf5"
CHUNK_SIZE = 100
ENCODE_TIMES = 20
DECODE_TIMES = 1
GENERATION_N = 20000

def reconstruction(model, XTE):
    print('Start reconstruction')
    # for every row (smile), it contains all the decoded smiles (smiles). len(XTE)*(ENCODE_TIMES*DECODE_TIMES)
    decode_result = []

    # for every group of smile
    for start_chunk in range(0, len(XTE), CHUNK_SIZE):
        smiles = XTE[start_chunk:start_chunk + CHUNK_SIZE]
        chunk_result = [[] for _ in range(len(smiles))]
        # for every encoding times
        for nEncode in range(ENCODE_TIMES):
            print('Current encode %d/%d ' % ( nEncode + 1, ENCODE_TIMES))
            # z: encoded latent points
            # NOTE: this operation returns the mean of the encoding distribution
            # if you would like it to sample from that distribution instead
            # replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
            z1 = model.encode(smiles)

            # for every decoding times
            for nDecode in range(DECODE_TIMES):
                resultSmiles = model.decode(z1)
                for index, s in enumerate(resultSmiles):
                    chunk_result[index].append(s)
        # add to decode_result
        decode_result.extend(chunk_result)

    assert len(decode_result) == len(XTE)
    return np.array(decode_result)

def prior(model):
    print('Start prior')
    # for every row (smile), it contains all the decoded smiles (smiles). len(XTE)*(ENCODE_TIMES*DECODE_TIMES)

    # sampling pz
    priors = []
    for i in range(1000):
        priors.append(np.random.normal(0, 1, 56))

    # for every group of smile
    decode_result = []
    for start_chunk in range(0, len(priors), CHUNK_SIZE):
        smiles = np.array(priors[start_chunk:start_chunk + CHUNK_SIZE])
        chunk_result = [[] for _ in range(len(smiles))]

        # for every decoding times
        for nDecode in range(DECODE_TIMES):
            resultSmiles = model.decode(smiles)
            for index, s in enumerate(resultSmiles):
                chunk_result[index].append(s)

        # add to decode_result
        decode_result.extend(chunk_result)

    assert len(decode_result) == len(priors)
    return np.array(decode_result)


def generation(model):
    print('generation')
    priors = []
    for i in range(GENERATION_N):
        priors.append(np.random.normal(0, 1, 56))

    # for every group of smile
    decode_result = [[]]
    for start_chunk in range(0, len(priors), CHUNK_SIZE):
        smiles = np.array(priors[start_chunk:start_chunk + CHUNK_SIZE])

        resultSmiles = model.decode(smiles)
        for index, s in enumerate(resultSmiles):
            decode_result[0].append(s)

    return np.array(decode_result)




def main():
    decoded_file = GRAMMAR_WEIGHTS.split(".")[0] + "_decRes.txt"
    priors_file = GRAMMAR_WEIGHTS.split(".")[0] + "_priorsRes.txt"
    generation_file = GRAMMAR_WEIGHTS.split(".")[0] + "_generationRes.txt"
    grammar_model = molecule_vae.Qm9GrammarModel(GRAMMAR_WEIGHTS)


    D = read_qm9()
     #fix problem about molecule with '.' inside
    XTE = []
    for mol in D:
        if "." not in mol:
            XTE.append(mol)
    XTE = XTE[0:5000]
    decoded_result = reconstruction(grammar_model, XTE)
    save_decoded_results(XTE, decoded_result, decoded_file)
    # decoded_priors = prior(grammar_model)
    # save_decoded_priors(decoded_priors, priors_file)
    decoded_generation = generation(grammar_model)
    save_decoded_priors(decoded_generation, generation_file)




if __name__ == '__main__':
    main()
