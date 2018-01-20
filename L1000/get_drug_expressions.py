from cmapPy.pandasGEXpress import parse
import csv
import json

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# Roadmap:
# 1. get list of all the perts we can get descriptors for
drug_descriptors_dict = {}
with open("/data/datasets/gwoo/L1000/LDS-1191/WorkingData/X_all_descriptors.tab", "r") as tab_file:
    reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
    drug_descriptors_dict = dict((rows[1],rows[0]) for rows in reader)

# 2. get all the batch ids with their pert ids
drug_dict = {}
with open("/data/datasets/gwoo/L1000/LDS-1191/Metadata/Small_Molecule_Metadata.txt", "r") as tab_file:
    reader = csv.reader(tab_file, dialect='excel', delimiter='\t')
    drug_dict = dict((rows[9],rows[8]) for rows in reader)

# 3 load expressions data
lm_genes = json.load(open('one_landmark_gene.json'))
lm_gene_entrez_ids = []
for lm_gene in lm_genes:
    lm_gene_entrez_ids.append(lm_gene['entrez_id'])
# print(lm_gene_entrez_ids)

level_5_gctoo = parse(
    "/home/gwoo/Data/L1000/LDS-1191/Data/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
    col_meta_only=False, row_meta_only=False, rid=lm_gene_entrez_ids)

# print(level_5_gctoo.col_metadata_df.index)
# print(level_5_gctoo.row_metadata_df.index)

# 	get an array with the drug id
drug_id_expressions = []

# # add headers
# row = []
# row.append('drug_id')
# for gene_id in lm_gene_entrez_ids:
#     row.append(gene_id)
# drug_id_expressions.append(row)

length = len(level_5_gctoo.col_metadata_df.index)
# length = 5

# 3. For every experiment
for i in range(length):
    print('Processing ' + str(i) + ' of ' + str(length) )
    col_name = level_5_gctoo.col_metadata_df.index[i]
    # parse the drug id here
    start = find_nth(col_name, ":", 1)
    end = find_nth(col_name, ":", 2)
    batch_id = col_name[start+1:end]
    if batch_id in drug_dict:
        drug_id = drug_dict[batch_id]
        if drug_id in drug_descriptors_dict:
            column = level_5_gctoo.data_df[col_name]
            row = []
            row.append(drug_id)
            for gene_id in lm_gene_entrez_ids:
                row.append(column[gene_id])
            drug_id_expressions.append(row)

# 3. save the list to csv
with open("/home/gwoo/Data/L1000/LDS-1191/WorkingData/Y_drug_id_one_expression.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(drug_id_expressions)

