from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv, get_trimmed_feature_dict
from scipy.spatial.distance import mahalanobis
import numpy as np
import matplotlib.pyplot as plt
import datetime

# x = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 1]])
# cov = np.cov(x.T)
# print(cov.shape)
# IV = np.linalg.inv(cov)

# remove duplicate drugs
def remove_dups(dict):
    unique_dict = {}
    check_dups = {}
    for drug_id in dict:
        value = drug_features_dict[drug_id]
        val_str = ''.join(value)
        if val_str in check_dups:
            # print (drug_id)
            continue
        check_dups[val_str] = 1

        drug_features = np.asarray(value, dtype='float16')
        unique_dict[drug_id] = drug_features
    return unique_dict

def get_corr_cols(a):
    uneeded_cols = []
    n_cols = a.shape[1]
    zero_cols = []
    for col1 in range(0, n_cols):
        col1a = a[:, col1]
        sum = np.sum(abs(col1a))
        if sum == 0:
            print(col1, "is zero")
            zero_cols.append(col1)
            uneeded_cols.append(col1)

    for col1 in range(0, n_cols):
        col1a = a[:, col1]
        suma = np.sum(col1a)
        # print("doing col", col1)
        if suma == 0:
            continue
        for col2 in range(col1, n_cols):
            if col1 == col2:
                continue
            col1b = a[:, col2]
            sumb = np.sum(col1b)
            if sumb == 0:
                continue
            corr = np.corrcoef(col1a, col1b)
            if corr[0][1] == 1 or corr[0][1] == -1:
                print("correlation in", col1, col2, corr)
                uneeded_cols.append(col2)
    return uneeded_cols

def get_array(dict):
    list = []
    for key in dict:
        value = dict[key]
        list.append(value)

    return np.array(list)

def get_distances():
    drug_features_dict = get_feature_dict('data/smiles_rdkit_maccs.csv') #, use_int=True)
    unique_drug_features_dict = remove_dups(drug_features_dict)
    drug_features = get_array(unique_drug_features_dict)
    # one_feature = drug_features[0]
    # print(drug_features.shape)
    # uneeded_cols = get_corr_cols(drug_features)
    uneeded_cols = [0,1,2,4,5,6,7,59,65,113,135,144,143]
    # print(len(drug_features_dict))

    n_cols = drug_features.shape[1]
    cols = range(0, n_cols)
    cols = [x for x in cols if x not in uneeded_cols]

    x = drug_features[:, cols]
    print(x.shape)
    cov = np.cov(x.T)
    # print(cov.shape)
    IV = np.linalg.inv(cov)

    distances = []
    n = x.shape[0]
    lapse = 0
    time = datetime.datetime.now()
    for source_i in range(0, n):
        source = x[source_i]
        if source_i % 20 == 0:
            newtime = datetime.datetime.now()
            lapse = newtime - time
            time = newtime
            print(str(time), "getting distances for molecule", source_i, lapse)
        for target_i in range(source_i, n):
            if source_i == target_i:
                continue
            target = x[target_i]
            distances.append(mahalanobis(source, target, IV))

    np.savez("/data/datasets/gwoo/L1000/LDS-1191/Output/appDomain/distances", distances)
    plt.hist(distances, bins=100)
    plt.show()

def plot_hist():
    import seaborn as sns
    distances = np.load("/data/datasets/gwoo/L1000/LDS-1191/Output/appDomain/distances.npz")
    # distances = {}
    # distances['arr_0'] = np.random.normal(size=100)
    plot = sns.distplot(distances['arr_0'], bins=100, axlabel="Mahalanobis Distance")
    plot.set_title("Histogram of Distances between Molecules")
    plot.set(ylabel="Normalized Count")
    plot.ticklabel_format(style='plain') #, axis='both', scilimits=(0, 0))

    # x = np.random.normal(size=100)
    # plot = sns.distplot(x)
    fig = plot.get_figure()
    fig.show()
    fig.savefig("/data/datasets/gwoo/L1000/LDS-1191/Output/appDomain/plot.png")

# get_distances()
plot_hist()

