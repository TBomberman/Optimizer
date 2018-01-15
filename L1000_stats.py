from cmapPy.pandasGEXpress import parse
import json
import stats_analysis
import os
import pandas as pd

def load_landmark_genes():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'L1000/one_landmark_gene.json')

    lm_genes = json.load(open(file_path))
    ids = []
    for lm_gene in lm_genes:
        ids.append(lm_gene['entrez_id'])
    return ids

def load_gene_expression_data():
    return parse(
        "/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        # col_meta_only=False, row_meta_only=False, rid=['3638'])
        col_meta_only=False, row_meta_only=False, rid=lm_gene_entrez_ids)

lm_gene_entrez_ids = load_landmark_genes()
level_5_gctoo = load_gene_expression_data()
length = len(level_5_gctoo.col_metadata_df.index)

for gene_id in lm_gene_entrez_ids:
    one_gene_expression_values = []
    for i in range(length):
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        one_gene_expression_values.append(column[gene_id])

stats_analysis.get_stats(one_gene_expression_values)