import os
os.environ["PATH"] += r";C:\ProgramData\Anaconda3\envs\my-rdkit-env"
os.environ["PATH"] += r";C:\ProgramData\Anaconda3\envs\my-rdkit-env\Library\mingw-w64\bin"
os.environ["PATH"] += r";C:\ProgramData\Anaconda3\envs\my-rdkit-env\Library\usr\bin"
os.environ["PATH"] += r";C:\ProgramData\Anaconda3\envs\my-rdkit-env\Library\bin"
os.environ["PATH"] += r";C:\ProgramData\Anaconda3\envs\my-rdkit-env\Script"
from L1000.data_loader import get_feature_dict, load_csv
from rdkit.Chem import Descriptors, MACCSkeys
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem

i=0
finger_dimension = 256
molecules = []
fps = []
id = []
smiles = []
names = []

import os
path = os.path.dirname(os.path.abspath(__file__))
print(path)
# drug_dict = get_feature_dict('G:/GodwinWoo/LINCS/LDS-1191/Metadata/GSE92742_Broad_LINCS_pert_info.txt',
#                              delimiter='\t', use_int=False)
drug_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_pert_info.txt',
                             delimiter='\t', use_int=False)

count = 0
for key in drug_dict:
    count += 1
    try:
        smiles = drug_dict[key][5]
        m = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(m)
        molecules.append(m)
        # fps.append(np.array([fp for fp in MACCSkeys.GenMACCSKeys(m).ToBitString()]))
        fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=finger_dimension), fp)
        fps.append(fp)
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
np.savetxt("data/smiles_rdkit_morgan.csv", data_header, delimiter=",", fmt="%s")
