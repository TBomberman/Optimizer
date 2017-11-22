import csv

# 1. get pert_info smiles
drug_dict1 = {}
with open("/data/datasets/gwoo/L1000/LDS-1191/Metadata/GSE92742_Broad_LINCS_pert_info.txt", "r") as tab_file:
    reader1 = csv.reader(tab_file, dialect='excel', delimiter='\t')
    drug_dict1 = dict((rows[0],rows[6]) for rows in reader1 if (rows[6] != '-666'))

# 2. get sm smiles
drug_dict2 = {}
with open("/data/datasets/gwoo/L1000/LDS-1191/Metadata/Small_Molecule_Metadata.txt", "r") as tab_file:
    reader2 = csv.reader(tab_file, dialect='excel', delimiter='\t')
    drug_dict2 = dict((rows[8], rows[6]) for rows in reader2)

# 3. merge them
drug_dict = {}
for drug_id in drug_dict1:
    if drug_id not in drug_dict and drug_id != 'BRD-K56060837': # BRD-K56060837 smiles was causing exception in dragon
        drug_dict[drug_id] = drug_dict1[drug_id]

for drug_id in drug_dict2:
    if drug_id not in drug_dict and drug_id != 'sm_center_canonical_id' and drug_id != 'BRD-K56060837':
        drug_dict[drug_id] = drug_dict2[drug_id]

print(len(drug_dict1))
print(len(drug_dict2))
print(len(drug_dict))

with open("/home/gwoo/Data/L1000/LDS-1191/WorkingData/merged_id_smiles.smi", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for key, value in drug_dict.items():
        writer.writerow([value, key])
