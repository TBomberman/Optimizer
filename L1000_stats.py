import json
import stats_analysis
import os
from L1000.data_loader import load_gene_expression_data

def load_landmark_genes():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'L1000/one_landmark_gene.json')

    lm_genes = json.load(open(file_path))
    ids = []
    for lm_gene in lm_genes:
        ids.append(lm_gene['entrez_id'])
    return ids

lm_gene_entrez_ids = load_landmark_genes()
level_5_gctoo = load_gene_expression_data(lm_gene_entrez_ids)
length = len(level_5_gctoo.col_metadata_df.index)

for gene_id in lm_gene_entrez_ids:
    one_gene_expression_values = []
    for i in range(length):
        col_name = level_5_gctoo.col_metadata_df.index[i]
        column = level_5_gctoo.data_df[col_name]
        one_gene_expression_values.append(column[gene_id])

stats_analysis.get_stats(one_gene_expression_values)