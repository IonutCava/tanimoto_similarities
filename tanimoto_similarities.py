import time
import random
import sys
import seaborn as sns
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.Chem.Draw import SimilarityMaps

print("Copyright (c) 2024 Ionut Cava - MIT License")

#Permission is hereby granted, free of charge, to any person 
#obtaining a copy of this software and associated documentation
#files (the "Software"), to deal in the Software without
#restriction, including without limitation the rights to use, copy,
#modify, merge, publish, distribute, sublicense, and/or sell copies
#of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be
#included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
#BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
#ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

parser = argparse.ArgumentParser("tanimoto_similarities")
parser.add_argument("--fingerprint", help="The fingerprinting method to use (e.g. fcfc4, avalon, lfcfp6, etc)", default="fcfc4", type=str)
parser.add_argument("--limit", help="Process only the first n entries from the source file", type=int, required=False, default=sys.maxsize)
args = parser.parse_args()

fpdict = {}
fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpdict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict['ap'] = lambda m: Pairs.GetAtomPairFingerprint(m)
fpdict['tt'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpdict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)

if not args.fingerprint in fpdict.keys():
    print("Invalid fingerprinting method!")
    sys.exit("implemented fingerprints: \n - ecfc0, ecfp0, maccs \n - ap, apbv, tt \n - hashap, hashtt --> with 1024 bits \n - ecfp4, ecfp6, ecfc4, ecfc6 -> with 1024 bits \n - fcfp4, fcfp6, fcfc4, fcfc6 --> with 1024 bits \n - avalon --> with 1024 bits \n - laval --> with 16384 bits \n - lecfp4, lecfp6, lfcfp4, lfcfp6 --> with 16384 bits \n - rdk5, rdk6, rdk7")

np.set_printoptions(threshold=sys.maxsize)
ligands_df = pd.read_csv("smiles.csv" , index_col=0 )

molecules = []
labels = []

for _, smiles in ligands_df[[ "SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))

for _, names in ligands_df[[ "Ligand_name"]].itertuples():
    labels.append(names)

count = min(len(molecules), args.limit)

mfpgen = fpdict[args.fingerprint]

fgrps = [ mfpgen(mol) for mol in molecules[:count]]

nfgrps = len(fgrps)
print("Number of fingerprints:", nfgrps)

similarities = np.zeros((nfgrps, nfgrps))

for i in range(1, nfgrps):
    similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
    similarities[i, :i] = similarity
    similarities[:i, i] = similarity

sns.set(font_scale=0.8)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(similarities, dtype=bool)
mask[np.triu_indices_from(mask)] = True

tri_lower_diag = np.tril(similarities, k=0)

fig, ax = plt.subplots(figsize=(12,10))
fig.canvas.manager.set_window_title('Tanimoto Calculator')
plot = sns.heatmap(tri_lower_diag, annot = True, annot_kws={"fontsize":5}, mask=mask, cmap=cmap, center=0,
                   square=True, xticklabels=labels[:count], yticklabels=labels[:count], linewidths=.7, cbar_kws={"shrink": .5})
plt.title(f'Tanimoto similarity using \'{args.fingerprint}\' fingerprinting', fontsize = 18)
plt.show()

fig = plot.get_figure()
fig.savefig("tanimoto_heatmap.png") 
