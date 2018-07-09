from L1000.data_loader import get_feature_dict, load_csv
from rdkit.Chem import Descriptors, MACCSkeys
import numpy as np
from rdkit import Chem, DataStructs

i=0
finger_dimension = 167
molecules = []
fps = []
id = []
smiles = []
names = []

import os
path = os.path.dirname(os.path.abspath(__file__))
print(path)
drug_dict = get_feature_dict('data/GermanTop10SOX2.csv', delimiter=',', use_int=False)

for key in drug_dict:
    try:
        m = Chem.MolFromSmiles(drug_dict[key][0])
        Chem.Kekulize(m)
        molecules.append(m)
        fps.append(np.array([fp for fp in MACCSkeys.GenMACCSKeys(m).ToBitString()]))
        id.append(key)
    except:
        # fps.append(np.repeat(0, finger_dimension))
        print(i, key, m)
    i += 1

header = ["mol"]
for i in range(finger_dimension):
    header.append("fps"+ str(i))

fps = np.array(fps).reshape(len(fps),finger_dimension)
id = np.array(id)    
id = id.reshape(len(fps),1)
data = np.hstack((id,fps))
header = np.array(header).reshape(1,len(header))
data_header = np.vstack((header,data))
np.savetxt("data/german_smiles_rdkit_maccs.csv", data_header, delimiter=",", fmt="%s")
