from L1000.gene_predictor import Gene_Predictor
import csv
import numpy as np
import helpers.email_notifier as en

def get_predictions(train_data, train_labels):
    model = Gene_Predictor()
    model.train(train_data, train_labels)

    file_name = '/data/datasets/gwoo/tox21/ZincCompounds_WaitOK_maccs_full.csv'
    with open(file_name, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=',')
        next(reader)
        counter = 0
        for row in reader:
            if counter % 10000 == 0:
                print("Evaluating molecule #", counter)
            fingerprint = []
            for value in row[1:]:
                fingerprint.append(int(value))
            molecule_id = row[0]
            data = np.asarray([fingerprint])
            prediction = model.predict(data)
            if prediction[0][0] > prediction[0][1]:
                print(molecule_id, "Downreglate")
                en.notify("found compound" + molecule_id)
            counter += 1


