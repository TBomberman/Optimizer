import csv, json

tca_up_dict = {}
tca_down_dict = {}
vpc_up = []
vpc_down = []

with open("data/PCAExpression.csv", "r") as csv_file:
    reader = csv.reader(csv_file, dialect='excel', delimiter=',')
    next(reader)
    for row in reader:
        test = row[0]
        if row[1] != '':
            tca_up_dict[row[1]] = row[0]
        if row[4] != '':
            tca_down_dict[row[4]] = row[3]
        if row[2] != '':
            vpc_up.append(row[2])
        if row[5] != '':
            vpc_down.append(row[5])

concensus_up_ids = []
for gene in vpc_up:
    if gene in tca_up_dict:
        concensus_up_ids.append(tca_up_dict[gene])
        # print(tca_up_dict[gene])

concensus_down_ids = []
for gene in vpc_down:
    if gene in tca_down_dict:
        concensus_down_ids.append(tca_down_dict[gene])
        # print(tca_down_dict[gene])

# check if they are in the landmark genes
def get_gene_id_dict():
    lm_genes = json.load(open('data/landmark_genes.json'))
    dict = {}
    for lm_gene in lm_genes:
        dict[lm_gene['entrez_id']] = lm_gene['gene_symbol']
    return dict

gene_dict = get_gene_id_dict()

print("Overexpressed landmark gene ids:")
for up_gene in concensus_up_ids:
    if up_gene in gene_dict:
        print(up_gene)

print("\nUnderexpressed landmark gene ids:")
for down_gene in concensus_down_ids:
    if down_gene in gene_dict:
        print(down_gene)

