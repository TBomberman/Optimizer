# run this file locally with anaconda with rdkit and also openeye installed

import os
from rdkit.Chem import Descriptors, MACCSkeys
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem
import csv

def load_csv(file):
    # load data
    expression = []
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel')
        for row in reader:
            expression.append(row)
    return expression

def get_feature_dict(file, delimiter=',', key_index=0, use_int=False):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        if use_int:
            my_dict = {}
            for row in reader:
                list = []
                for value in row[1:]:
                    list.append(int(value))
                my_dict[row[key_index]] = list
            return my_dict
        return dict((row[key_index], row[1:]) for row in reader)
i=0
finger_dimension = 2048
molecules = []
fps = []
id = []
smiles = []
names = []

import os
path = os.path.dirname(os.path.abspath(__file__))
print(path)
drug_dict = get_feature_dict('G:/GodwinWoo/LINCS/LDS-1191/Metadata/GSE92742_Broad_LINCS_pert_info.txt',
                             delimiter='\t', use_int=False)
# this is the LNCAP dataset
# drug_dict14 = get_feature_dict('G:/GodwinWoo/LINCS/LDS-1484/Metadata/GSE70138_Broad_LINCS_pert_info.txt',
#                              delimiter='\t', use_int=False)
# drug_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_pert_info.txt',
#                              delimiter='\t', use_int=False)
# drug_dict = {}
# drug_dict['EnzoLincs'] = ['','','','','','CNC(=O)c1ccc(cc1F)N1C(=S)N(C(=O)C1(C)C)c1ccc(C#N)c(c1)C(F)(F)F']
# drug_dict['EnzaPubChem'] = ['','','','','','CC1(C(=O)N(C(=S)N1C2=CC(=C(C=C2)C(=O)NC)F)C3=CC(=C(C=C3)C#N)C(F)(F)F)C']
# drug_dict['EnzaCanon'] = ['','','','','','CNC(=O)C1=C(F)C=C(C=C1)N2C(=S)N(C(=O)C2(C)C)C3=CC(=C(C=C3)C#N)C(F)(F)F']
# drug_dict['Buclutomide'] = ['','','','','','CC(O)(C[S](=O)(=O)C1=CC=C(F)C=C1)C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F']
# drug_dict['Apolutamide'] = ['','','','','','CNC(=O)C1=C(F)C=C(C=C1)N2C(=S)N(C(=O)C23CCC3)C4=CN=C(C#N)C(=C4)C(F)(F)F']
# drug_dict['Darolutamide'] = ['','','','','','CC(O)C1=CC(=N[NH]1)C(=O)NC(C)C[N]2C=CC(=N2)C3=CC(=C(C=C3)C#N)Cl']
# drug_dict['VPC13789'] = ['','','','','','CC(C)NC(=O)C1=CC=CC2=NC(=CC=C12)C3=C[NH]C4=C(F)C(=C(F)C=C34)F']
# drug_dict['VPC14449'] = ['','','','','','BrC1=C(Br)[N](C=N1)C2=CSC(=N2)N3CCOCC3']
# drug_dict['VPC17005'] = ['','','','','','O=C(NC1=NCCS1)C2=CSC3=CC=CC=C23']
# drug_dict['VPC220062'] = ['','','','','','OC(CCl)COC1=CC=C(Br)C2=C(OCC(O)CCl)C=CC=C12']
# drug_dict['ZINC335870/VPC2055'] = ['','','','','','O[C@H](CCl)CNC1=C2C=CC=C(NC[C@@H](O)CCl)C2=CC=C1']
# drug_dict['ZINC145981020/VPC220010'] = ['','','','','','O[C@@H](CCl)COC1=C2C=CC=C(OC[C@@H](O)CCl)C2=CC=C1']

# nathan's drugs
drug_dict = {}
drug_dict['Enzalutamide'] = ['','','','','','CNC(=O)C1=C(F)C=C(C=C1)N1C(=S)N(C(=O)C1(C)C)C1=CC=C(C#N)C(=C1)C(F)(F)F']
drug_dict['VPC13789'] = ['','','','','','Fc1c(F)c2[nH]cc(-c3nc4c(c(C(=O)NC(C)C)ccc4)cc3)c2cc1F']
drug_dict['VPC14449'] = ['','','','','','Brc1n(-c2nc(N3CCOCC3)sc2)cc(Br)n1']
drug_dict['VPC17005'] = ['','','','','','O=C(NC=1SCCN=1)c1c2c(sc1)cccc2']
drug_dict['VPC220010'] = ['','','','','','ClCC(O)COc1c2c(c(OCC(O)CCl)ccc2)ccc1']

count = 0
for key in drug_dict:
    count += 1
    try:
        smiles = drug_dict[key][5]
        m = Chem.MolFromSmiles(smiles)
        # Chem.Kekulize(m) # this generates different fps from canonical smiles
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
np.savetxt("nathans_morgan_2048_nk.csv", data_header, delimiter=",", fmt="%s")
