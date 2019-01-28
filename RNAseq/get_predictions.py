import numpy as np
from pathlib import Path
from L1000.gene_predictor import load_model
from L1000.data_loader import get_feature_dict

# load model
def load_model_from_file_prefix(model_file_prefix):
    model_file = Path(model_file_prefix + ".json")
    if not model_file.is_file():
        print(model_file.name + "File not found")
    return load_model(model_file_prefix)


def predict_nathans():
    up_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/LNCAP_NK_LM_Up"
    up_model = load_model_from_file_prefix(up_model_filename_prefix)
    down_model_filename_prefix = "/data/datasets/gwoo/L1000/LDS-1191/saved_models/screen_ar/LNCAP_NK_LM_Down"
    down_model = load_model_from_file_prefix(down_model_filename_prefix)

    gene_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/gene_go_fingerprint_moreThan3.csv')
    drug_features_dict = get_feature_dict(
        '/data/datasets/gwoo/Python/Optimizer/L1000/LDS-1191/data/nathans_morgan_2048_nk.csv')

    # target_gene_features_dict = {
    #     'AR': gene_features_dict['AR'],
    #     'KLK3': gene_features_dict['KLK3'],
    #     'KLK2': gene_features_dict['KLK2'],
    #     'TMPRSS2': gene_features_dict['TMPRSS2'],
    #     'CDC20': gene_features_dict['CDC20'],
    #     'CDK1': gene_features_dict['CDK1'],
    #     'CCNA2': gene_features_dict['CCNA2'],
    #     'UBE2C': gene_features_dict['UBE2C'],
    #     'AKT1': gene_features_dict['AKT1'],
    #     'UGT2B15': gene_features_dict['UGT2B15'],
    #     'UGT2B17': gene_features_dict['UGT2B17'],
    #     'TRIB1': gene_features_dict['TRIB1']
    # }

    data = []
    descriptions = []
    nates_missing_genes = [
        'GATA3',
        'RPL39L',
        'IKZF1',
        'CXCL2',
        'HMGA2',
        'TLR4',
        'SPP1',
        'MEF2C',
        'PRKCQ',
        'MMP1',
        'PTGS2',
        'ICAM3',
        'INPP1',
        # 'KIT',  # not in 2reps
        # 'COL4A1',  # not in 2reps
        # 'GNA15',  # not in 2reps
        # 'SERPINE1',  # not in 2reps
        # 'SNAP25',  # not in 2reps
        # 'SOX2',  # not in 2reps
        # 'MMP2',  # not in 2reps
        # 'ICAM1',  # not in 2reps
    ]
    for gene in nates_missing_genes:
        gene_features_dict.pop(gene, None)
    drug_features_dict.pop('Enzalutamide', None)
    for drug in drug_features_dict:
        for gene in gene_features_dict:
            data.append(drug_features_dict[drug] + gene_features_dict[gene])
            descriptions.append(drug + ", " + gene)
    data = np.asarray(data, dtype=np.float16)

    up_predictions = up_model.predict(data)
    down_predictions = down_model.predict(data)

    # for i in range(0, len(data)):
    #     up_prediction = up_predictions[i]
    #     if up_prediction[1] > 0.561:  # max f cutoff
    #         # print(descriptions[i], "Up Probability", up_prediction[1])
    #         print(descriptions[i], ", Up,", 1)
    #     else:
    #         print(descriptions[i], ", Up,", 0)
    # for i in range(0, len(data)):
    #     down_prediction = down_predictions[i]
    #     if down_prediction[1] > 0.648:  # max f cutoff
    #         # print(descriptions[i], "Down Probability", down_prediction[1])
    #         print(descriptions[i], ", Down,", 1)
    #     else:
    #         print(descriptions[i], ", Down,", 0)
    return up_predictions, down_predictions, drug_features_dict, gene_features_dict

# predict_nathans()
