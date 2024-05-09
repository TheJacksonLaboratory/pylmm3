import numpy as np
import os
from pylmm.lmm import GWAS  # Ensure this module is the same used in Python 2 but compatible with Python 3

# I generated this data by using the python 2 pylmm library.
data_path = ".local/gwas/"

Y=np.load(os.path.join(data_path, 'Y.npy'))
X=np.load(os.path.join(data_path, 'X.npy'))
K=np.load(os.path.join(data_path, 'K.npy'))
Kva=np.load(os.path.join(data_path, 'Kva.npy'))
Kve=np.load(os.path.join(data_path,'Kve.npy'))
T_stats_old=np.load(os.path.join(data_path,'T_stats.npy'))
P_values_old=np.load(os.path.join(data_path,'P_values.npy'))

# Run GWAS again in Python 3 to generate new T_stats and P_values
T_stats_new, P_values_new = GWAS(Y, X, K, Kva, Kve)

# Compare
print("Comparison of T-stats:", np.allclose(T_stats_old, T_stats_new, equal_nan=True))
print("Comparison of P-values:", np.allclose(P_values_old, P_values_new, equal_nan=True))