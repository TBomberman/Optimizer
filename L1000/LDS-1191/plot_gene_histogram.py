import json
import matplotlib.pyplot as plt
import numpy as np
from L1000.data_loader import load_gene_expression_data, load_csv, printProgressBar, get_feature_dict

# go through the data set
# for each gene, calculate the variance
# store the top gene and print it out

# load expressions data
def load_landmark_genes():
    symbols = {}
    lm_genes = json.load(open('data/landmark_genes.json'))
    for lm_gene in lm_genes:
        symbols[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return symbols

symbols = load_landmark_genes()
gene_count_data_limit = 978
lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
lm_gene_entrez_ids = []
for sublist in lm_gene_entrez_ids_list :
    for item in sublist:
        lm_gene_entrez_ids.append(item)
# lm_gene_entrez_ids = ['2778']
level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)


def get_all_perts():
    length = len(level_5_gctoo.col_metadata_df.index)
    # length = 10
    one_gene_expression_values = []

    # For every gene
    for i in range(length):
        printProgressBar(i, length, prefix='Load experiments progress')
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        for gene_id in lm_gene_entrez_ids:
            one_gene_expression_values.append(column[gene_id])
    return one_gene_expression_values

def get_stats():
    values = get_all_perts()
    min = np.min(values)
    max = np.max(values)
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    print('min', min)
    print('max', max)
    print('mean', mean)
    print('median', median)
    print('std', std)

    q5 = np.percentile(values, 5)
    q25 = np.percentile(values, 25)
    q50 = np.percentile(values, 50)
    q75 = np.percentile(values, 75)
    q95 = np.percentile(values, 95)

    print("5% quantile: {}\n25% quantile: {}\n50% quantile: {}\n75% quantile: {}\n95% quantile: {}\n".format(q5, q25, q50, q75, q95))

def get_histogram():
    values = get_all_perts()
    plt.title('Histogram of Perturbations')
    plt.ylabel('Number of expressions')
    plt.xlabel('Z-score')
    plt.hist(values, bins=1000)
    plt.show()

def get_drug_counts_per_cell_line():
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start + len(needle))
            n -= 1
        return start

    experiments_dose_dict = get_feature_dict('/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_sig_info.txt', '\t', 0)
    length = len(level_5_gctoo.col_metadata_df.index)
    # length = 10
    cell_line_perts = {}

    # For every gene
    for i in range(length):
        printProgressBar(i, length, prefix='Load experiments progress')
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]

        # get drug features
        col_name_key = col_name[2:-1]
        if col_name_key not in experiments_dose_dict:
            continue
        experiment_data = experiments_dose_dict[col_name_key]
        drug_id = experiment_data[0]

        if cell_name not in cell_line_perts:
            cell_line_perts[cell_name] = {}

        # for gene_id in lm_gene_entrez_ids:
        if drug_id not in cell_line_perts[cell_name]:
            cell_line_perts[cell_name][drug_id] = 1
        # cell_line_perts[cell_name][drug_id] += 1

    for cell_name in cell_line_perts:
        print(cell_name, 'drug count', len(cell_line_perts[cell_name]))

# get_drug_counts_per_cell_line()
get_stats()