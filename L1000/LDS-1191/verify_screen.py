# run locally
# Get list of all the files and attributes of cedar
# get list of all the files and attributes of graham
# Compare and print out what you want to keep and not
import os
import csv
from rdkit import Chem


def get_top_scores(n_sections=1, section=0):
    scores_path = "H:/Zinc Scores/ZINC_15_morgan_2048_2D_scores"
    files = os.listdir(scores_path)

    files_per_section = int(len(files) / n_sections)
    start = section * files_per_section
    end = start + files_per_section
    files = files[start:end]

    file_counter = 0
    for file in files:
        # counter += 1
        # if counter > 500:
        #     break
        try:
            with open(scores_path + "\\" + file, "r") as csv_file:
                reader = csv.reader(csv_file, dialect='excel')
                counter = 0
                for row in reader:
                    if len(row) < 2:
                        print(file, "missing data in one line")
                        break
                    score = float(row[1])
                    # scores.append(score)
                    if score > 0.415:
                        counter += 1
                        print(str(counter) + ',', row[0] + ',', row[1])
                file_counter += 1
                if file_counter % 100 == 0: # or file_counter > 600:
                    print('file', int(file_counter), ', x,', 'x')
        except:
            print(file, "had a file read exception")
            continue

    # plt.title('Histogram')
    # plt.ylabel('Number of molecules')
    # plt.xlabel('AR prediction score')
    # plt.hist(scores, bins=1000)
    # plt.show()


def smiles2sdf():
    # to get the sdf output for fafdrugs to test toxicity
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


def verify_screen():
    # cedar_files = os.scandir('H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores_cedar')
    # graham_files = os.scandir('H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores_graham')

    cedar_sizes = {}
    cedar_mods = {}
    with os.scandir('H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores_cedar') as cedar_files:
        for entry in cedar_files:
            info = entry.stat()
            cedar_sizes[entry.name] = info.st_size
            cedar_mods[entry.name] = info.st_mtime

    with os.scandir('H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores_graham') as graham_files:
        for entry in graham_files:
            info = entry.stat()
            cedar_size = cedar_sizes[entry.name]
            if cedar_size != info.st_size:
                print(entry.name, "size difference cedar", cedar_size, "graham", info.st_size)
            # cedar_mtime = cedar_mods[entry.name]
            # if cedar_mtime != info.st_mtime:
            #     print(entry.name, "time difference cedar", cedar_mtime, "graham", info.st_mtime)


# verify_screen()
get_top_scores(5, 4)
