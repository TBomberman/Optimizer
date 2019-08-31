import json
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
import numpy as np
from L1000.data_loader import load_gene_expression_data, load_csv, printProgressBar, get_feature_dict
import seaborn as sns

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
# lm_gene_entrez_ids_list = load_csv('data/genes_by_var.csv')[:gene_count_data_limit]
# lm_gene_entrez_ids = []
# for sublist in lm_gene_entrez_ids_list :
#     for item in sublist:
#         lm_gene_entrez_ids.append(item)
lm_gene_entrez_ids = ['2778']


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def get_all_perts(target_cell_name=None):
    level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", lm_gene_entrez_ids)
    length = len(level_5_gctoo.col_metadata_df.index)
    # length = 10
    one_gene_expression_values = []

    # For every gene
    for i in range(length):
        printProgressBar(i, length, prefix='Load experiments progress')
        col_name = level_5_gctoo.col_metadata_df.index[i]

        # parse the cell name
        start = find_nth(col_name, "_", 1)
        end = find_nth(col_name, "_", 2)
        cell_name = col_name[start + 1:end]
        if cell_name != target_cell_name and target_cell_name != None:
            continue

        column = level_5_gctoo.data_df[col_name]
        for gene_id in lm_gene_entrez_ids:
            one_gene_expression_values.append(column[gene_id])
    return one_gene_expression_values

def get_stats():
    for cell in ['A549']: #, 'MCF7', 'HT29', 'A375', 'VCAP', 'A549']:
        print(cell)
        values = get_all_perts(cell)
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
    q10 = np.percentile(values, 10)
    q25 = np.percentile(values, 25)
    q50 = np.percentile(values, 50)
    q75 = np.percentile(values, 75)
    q90 = np.percentile(values, 90)
    q95 = np.percentile(values, 95)

    print("5% quantile: {}\n10% quantile: {}\n25% quantile: {}\n50% quantile: {}\n75% quantile: {}\n90% quantile: {}\n95% quantile: {}\n".format(q5, q10, q25, q50, q75, q90, q95))

def get_histogram():
    values = get_all_perts()

    import seaborn as sns
    plot = sns.distplot(values, bins=1000, axlabel="Z-score", kde=False, norm_hist=False)
    plot.set_title("Histogram of gene: GNAS")
    plot.set(ylabel="Number of Perturbation Values")
    # plot.ticklabel_format(style='plain')  # , axis='both', scilimits=(0, 0))
    fig = plot.get_figure()
    fig.set_size_inches(6, 4)
    fig.show()
    # fig.savefig("/data/datasets/gwoo/L1000/LDS-1191/Output/appDomain/plot.png")

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

        # parse the dosage unit and value
        dose_unit = experiment_data[5]
        if dose_unit != 'µM':  # standardize dose amounts
            # column counts: -666 17071, % 2833, uL 238987, uM 205066, ng 1439, ng / uL 2633, ng / mL 5625
            continue
        dose_amt = float(experiment_data[4])
        bin = 10
        if dose_amt < bin - 0.1 or dose_amt > bin + 0.1:  # only use the 5 mm bin
            continue

        if cell_name not in cell_line_perts:
            cell_line_perts[cell_name] = []

        cell_line_perts[cell_name].append(drug_id)

    for cell_name in cell_line_perts:
        length = len(list(set(cell_line_perts[cell_name])))
        print(cell_name, 'drug count', length)

def plot_normal():
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import math
    from matplotlib import collections  as mc
    import pylab as pl

    fig_size = plt.rcParams["figure.figsize"]
    # print("Current size:", fig_size)
    fig_size[1] = 3
    plt.cla()
    plt.rcParams["figure.figsize"] = fig_size

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, mlab.normpdf(x, mu, sigma))

    # percentiles
    vals = mlab.prctile(x, p=(10, 90))

    downx = vals[0]
    upx = vals[1]

    plt.plot(*[(downx, downx), (0, mlab.normpdf(downx, mu, sigma))], color='red')
    plt.plot(*[(downx, -3), (0, 0)], color='red', label='Downregulation')
    plt.plot(*[(upx, upx), (0, mlab.normpdf(upx, mu, sigma))], color='green')
    plt.plot(*[(upx, 3), (0, 0)], color='green', label='Upregulation')
    plt.plot(*[(downx, upx), (0, 0)], color='blue')  #, label='No regulation')
    plt.legend()

    plt.title("Top and Bottom 10% Significance Levels")
    plt.xlabel("Z-score")
    plt.ylabel("Density")
    plt.show()

def get_concentration_data_for_histogram():
    csv = load_csv("/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_inst_info.csv")
    doses = []
    for row in csv:
        dose_unit = row[7]
        if dose_unit != 'um':
            continue
        doses.append(float(row[6]))

    plt.title('Drug Dose Histogram')
    axes = plt.gca()
    axes.set_xlim([0, 15])
    plt.ylabel('Number of Experiments')
    plt.xlabel('Drug Concentrations (µM)')
    plt.hist(doses, bins=2000)
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    # fig = plt.figure(figsize = (8, 6), dpi=600)
    # fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.show()

# get_drug_counts_per_cell_line()
# get_stats()
# plot_normal()
# get_concentration_data_for_histogram()
get_histogram()