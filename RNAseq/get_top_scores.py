import os
import csv
import matplotlib.pyplot as plt
# from rdkit import Chem

def get_top_scores():
    scores_path = 'H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores_cedar'
    files = os.listdir(scores_path)

    # scores = []

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
                        continue
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


def compare_graham_cedar():
    scores_path = 'H:\Zinc Scores\ZINC_15_morgan_2048_2D_scores'
    files = os.listdir(scores_path)


def testBiomart():
    from pybiomart import Dataset
    import json
    import math

    # get a dict from biomart of entrez and ensembl and name
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
    df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene'])
    biomart_dict = {}
    for index, row in df.iterrows():
        if math.isnan(row[2]):
            continue
        entrez = int(row[2])
        if str(entrez) in biomart_dict:
            biomart_dict[str(entrez)].append(row[0] + " " + row[1])
        else:
            biomart_dict[str(entrez)] = [row[0] + " " + row[1]]

    # get dict of nate's gene name and ensemls
    nate_dict = {}
    with open('data/GeneNameConversion_index.txt', "r") as csv_file:
        nate_reader = csv.reader(csv_file, dialect='excel', delimiter='\t')
        for row in nate_reader:
            nate_dict[row[2]] = row[1]

    # get list of landmark genes get the name and entrez
    # for each lm gene get the gene name,
    #   then print the biomart's name and ensembl from the entrez, nate's name and ensemble from the gene name
    lm_genes = json.load(open('../L1000/LDS-1191/data/lm_plus_ar_genes.json'))
    for lm_gene in lm_genes:
        lm_entrez = lm_gene['entrez_id']
        lm_symbol = lm_gene['gene_symbol']
        if lm_symbol not in nate_dict:
            print("nate missing lm_symbol", lm_symbol)
            continue
        nate_ens = nate_dict[lm_symbol]
        if lm_entrez not in biomart_dict:
            print("biomart missing entrez", lm_entrez)
            continue
        bio_list = biomart_dict[lm_entrez]
        symbol_found = False
        entrez_found = False
        for item in bio_list:
            bio_ens, bio_symbol = item.split()
            if lm_symbol == bio_symbol:
                symbol_found = True
            if bio_ens == nate_ens:
                entrez_found = True
        if not symbol_found:
            print("lm_symbol not found in biomart's list for ", lm_entrez)
        if not entrez_found:
            print("nate_ens not found in biomart's list for ", lm_symbol)

# smiless2sdf()
# get_top_scores()
testBiomart()