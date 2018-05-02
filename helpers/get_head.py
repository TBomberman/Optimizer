# This script just check ths head of data files

import csv

file_name = '/home/gwoo/Data/zinc/ZincCompounds_InStock_maccs.tab'

with open(file_name, "r") as csv_file:
    reader = csv.reader(csv_file, dialect='excel', delimiter=',')
    next(reader)
    drug_counter = 0
    for row in reader:
        if drug_counter >=5:
            break
        print(row)
        print(len(row))
        drug_counter += 1

