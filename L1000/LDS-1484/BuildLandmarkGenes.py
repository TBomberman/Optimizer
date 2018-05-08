from L1000.data_loader import load_csv, load_gene_expression_data
import json, requests

lm_gene_entrez_ids_list = load_csv('data/lm_gene_ids')

json_data = []

level_5_gctoo = load_gene_expression_data("/home/gwoo/Data/L1000/LDS-1484/Data/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328.gctx")
length = len(level_5_gctoo.col_metadata_df.index)
row_name = level_5_gctoo.row_metadata_df.index[0]

for gene_id in lm_gene_entrez_ids_list:
    url = 'https://api.clue.io/api/genes?filter={"where":{"entrez_id":"' + gene_id[0] + '"}}&user_key=a1a15891784b70af2d7f5368db2fc863'
    response = requests.get(url)
    data = json.loads(response.content)
    print("got json for", gene_id)
    json_data.append(data[0])

with open('data/landmark_genes.json', 'w') as outfile:
    json.dump(json_data, outfile)