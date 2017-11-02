import numpy as np
from optimizer import do_optimize

data = np.loadtxt('/home/gwoo/Documents/Data/png_openeye/tox21_deepchem_InductiveDescriptors_qsar_label_NR-AhR.csv', skiprows=1, delimiter=",", usecols = range(1,52))
do_optimize(data)
