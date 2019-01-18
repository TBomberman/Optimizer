import os
import csv
import matplotlib.pyplot as plt
from rdkit import Chem

def get_top_scores():
    scores_path = 'H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores'
    files = os.listdir(scores_path)

    # scores = []

    file_counter = 0
    counter = 0
    for file in files:
        # counter += 1
        # if counter > 500:
        #     break
        with open(scores_path + "\\" + file, "r") as csv_file:
            reader = csv.reader(csv_file, dialect='excel')
            for row in reader:
                score = float(row[1])
                # scores.append(score)
                if score > 0.415:
                    counter += 1
                    print(str(counter) + ',', row[0] + ',', row[1])
            file_counter += 1
            print('file', int(file_counter), ', x,', 'x')

    # plt.title('Histogram')
    # plt.ylabel('Number of molecules')
    # plt.xlabel('AR prediction score')
    # plt.hist(scores, bins=1000)
    # plt.show()


def smiless2sdf():
    smiless = []
    smiless.append('CCS(=O)(=O)c1cnc(O)c(C(=O)N[C@@H]2CCO[C@H](C)C2)c1')
    smiless.append('C=C1C[C@H]2CC(N[C@@H]3CC[C@@H](NC(C)=O)CC3)C[C@@H](C1)C2')
    smiless.append('Cc1oncc1CN[C@@H]1CN(C(=O)Cn2ccc(C(F)F)n2)C[C@H]1C')
    smiless.append('CN(CCOCCO)[C@@H]1CCN(C(=O)Cn2ccc(C(F)F)n2)C1')
    mols = []
    for smiles in smiless:
        m = Chem.MolFromSmiles(smiles)
        mols.append(m)
    w = Chem.SDWriter('set1.sdf')
    for m in mols:
        w.write(m)


smiless2sdf()
