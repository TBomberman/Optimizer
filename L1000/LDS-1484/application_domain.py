from L1000.data_loader import get_feature_dict, load_gene_expression_data, printProgressBar, load_csv, get_trimmed_feature_dict
from scipy.spatial.distance import mahalanobis
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import jaccard_similarity_score
import seaborn as sns

# x = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 1]])
# cov = np.cov(x.T)
# print(cov.shape)
# IV = np.linalg.inv(cov)

def remove_non_lncap(dict):
    lncap_drugs = load_csv('data/LNCAPdrugs.csv')
    new_dict = {}
    for key in dict:
        if [key] in lncap_drugs:
            new_dict[key] = dict[key]
    return new_dict


# remove duplicate drugs
def remove_dups(dict):
    unique_dict = {}
    check_dups = {}
    for drug_id in dict:
        value = dict[drug_id]
        val_str = ''.join(value)
        if val_str in check_dups:
            # print (drug_id)
            continue
        check_dups[val_str] = 1

        drug_features = np.asarray(value, dtype='float16')
        unique_dict[drug_id] = drug_features
    return unique_dict

def remove_dup_np(arr):
    checkdup = []
    unique_arr = []
    for row in arr:
        val_str = ''.join(str(int(x)) for x in row)
        if val_str in checkdup:
            continue
        checkdup.append(val_str)
        unique_arr.append(row)
    return np.asarray(unique_arr)


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
        if col1 in uneeded_cols:
            continue
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
                if col2 not in uneeded_cols:
                    uneeded_cols.append(col2)
    return uneeded_cols

def get_array(dict):
    list = []
    for key in dict:
        value = dict[key]
        list.append(value)

    return np.array(list)

def remove_corr_features(all_features):
    uneeded_cols = load_csv('data/LNCAPcorr_cols.csv')
    uneeded_cols_int = []
    for col in uneeded_cols:
        uneeded_cols_int.append(int(col[0]))
    # uneeded_cols = get_corr_cols(all_features)
    # for col in uneeded_cols:
    #     print(col)
    print(len(uneeded_cols))
    n_cols = all_features.shape[1]
    cols = range(0, n_cols)
    cols = [x for x in cols if x not in uneeded_cols_int]
    return all_features[:, cols]

def get_uncorr_drug_features(include_nates=True):
    # first get the compounds that are from the LINCS dataset
    drug_features_dict = get_feature_dict('data/LDS1484_compounds_morgan_2048_nk.csv')  # , use_int=True)
    drug_features_dict = remove_non_lncap(drug_features_dict)

    # add the compounds that were in RNAseq
    if include_nates:
        nates_drugs = get_feature_dict('/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/nathans_morgan_2048_nk.csv')
        nates_drugs.pop('VPC220010', None)
        nates_drugs.pop('VPC13789', None)
        for nate_drug in nates_drugs:
            drug_features_dict[nate_drug] = nates_drugs[nate_drug]

    unique_drug_features_dict = remove_dups(drug_features_dict)
    drug_features = get_array(unique_drug_features_dict)
    # one_feature = drug_features[0]
    # print(drug_features.shape)
    # uneeded_cols = get_corr_cols(drug_features)  # !! this takes a long time to calculate so save the values if you can
    # for col in uneeded_cols:
    #     print(col)
    no_cor_np = remove_corr_features(drug_features)
    return remove_dup_np(no_cor_np)

def get_distances():
    x = get_uncorr_drug_features()
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

    # np.savez("/data/datasets/gwoo/L1000/LDS-1191/Output/appDomain/distances", distances)
    plt.hist(distances, bins=100)
    plt.show()

def plot_hist():
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

def check_nathan_duplicates():
    drug_features_dict = get_feature_dict('data/smiles_rdkit_maccs.csv') #, use_int=True)
    print("size of original drugs", len(drug_features_dict))
    nathan_drug_features_dict = get_feature_dict('data/nathan_smiles_rdkit_maccs.csv')  # , use_int=True)
    for nathan_drug_id in nathan_drug_features_dict:
        print("checking for duplicates of", nathan_drug_id)
        nathan_maccs = nathan_drug_features_dict[nathan_drug_id]
        nathan_val_str = ''.join(nathan_maccs)

        for drug_id in drug_features_dict:
            l1000_maccs = drug_features_dict[drug_id]
            l1000_val_str = ''.join(l1000_maccs)
            val_str = ''.join(l1000_val_str)
            if val_str == nathan_val_str:
                print(nathan_drug_id, "is the same as", drug_id)

    print("done")

def centeroidnp(arr):
    length = arr.shape[0]
    sum = arr.sum(axis=0)
    return sum/length

def get_distance_from_centroid():
    drug_features = get_uncorr_drug_features()
    centroid = centeroidnp(drug_features)

    cov = np.cov(drug_features.T)
    IV = np.linalg.inv(cov)

    nathan_drug_features_dict = get_feature_dict('data/nathan_smiles_rdkit_maccs.csv')  # , use_int=True)
    for nathan_drug_id in nathan_drug_features_dict:
        nathan_maccs = nathan_drug_features_dict[nathan_drug_id]
        nathan_maccs = remove_corr_features(np.array(nathan_maccs, dtype='float16').reshape(1,167))
        distance = mahalanobis(nathan_maccs, centroid, IV)
        print("distance of", nathan_drug_id, "from centroid", distance)


def get_jaccard_score_of_nates_drug(drug_id, lincs_drugs):
    nates_drugs = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/nathans_morgan_2048_nk.csv')

    nates_drug = nates_drugs[drug_id]
    nates_drug = np.reshape(np.array(nates_drug, np.float16), (1, -1))
    nates_drug = remove_corr_features(nates_drug)
    scores = []
    for lincs_drug in lincs_drugs:
        score = jaccard_similarity_score(lincs_drug, nates_drug[0])
        scores.append(score)
    return np.mean(scores)


def get_jaccard_scores():
    lincs_drug_features = get_uncorr_drug_features(False)
    num_drugs = len(lincs_drug_features)

    # get the scores from each other
    scores = []
    for i in range(0, num_drugs):
        for j in range(i+1, num_drugs):
            source_drug = lincs_drug_features[i]
            target_drug = lincs_drug_features[j]
            score = jaccard_similarity_score(source_drug, target_drug)
            scores.append(score)

    plot = sns.distplot(scores, bins=100, axlabel="Jaccard Similarity Coefficient", norm_hist=False)
    plot.set_title("Histogram of Jaccard Similarity Scores Between Trained Compounds")
    plot.set(ylabel="Count")
    plot.ticklabel_format(style='plain')  # , axis='both', scilimits=(0, 0))

    score_enza = get_jaccard_score_of_nates_drug("Enzalutamide", lincs_drug_features)
    score_17005 = get_jaccard_score_of_nates_drug("VPC17005", lincs_drug_features)
    score_14449 = get_jaccard_score_of_nates_drug("VPC14449", lincs_drug_features)
    plt.plot([score_enza, score_enza], [0, 25], color="yellow", label="Enzalutamide")
    plt.plot([score_17005, score_17005], [0, 25], color="red", label="VPC-17005")
    plt.plot([score_14449, score_14449], [0, 25], color="blue", label="VPC-14449")

    plot.legend(loc="upper right")

    fig = plot.get_figure()
    fig.show()

    # plt.show()


# get_distances()
# plot_hist()
# check_nathan_duplicates()
# get_distance_from_centroid()
get_jaccard_scores()