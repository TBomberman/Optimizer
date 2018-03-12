from mlp_optimizer import do_optimize
import numpy as np

xfilename = '/data/datasets/gwoo/L1000/LDS-1191/WorkingData/x.csv'
yfilename = '/data/datasets/gwoo/L1000/LDS-1191/WorkingData/y.csv'

features = np.genfromtxt(xfilename)
labels = np.genfromtxt(yfilename)

do_optimize(2, features, labels)