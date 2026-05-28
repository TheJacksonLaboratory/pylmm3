import numpy as np
import os
from pylmm3.gwas import run_gwas

data_path = ".local/gwas/"

if not os.path.isdir(data_path):
    print("SKIP: test data not found at", data_path)
else:
    Y        = np.load(os.path.join(data_path, 'Y.npy'))
    X        = np.load(os.path.join(data_path, 'X.npy'))
    K        = np.load(os.path.join(data_path, 'K.npy'))
    Kva      = np.load(os.path.join(data_path, 'Kva.npy'))
    Kve      = np.load(os.path.join(data_path, 'Kve.npy'))
    T_stats_old  = np.load(os.path.join(data_path, 'T_stats.npy'))
    P_values_old = np.load(os.path.join(data_path, 'P_values.npy'))

    # Wrap the SNP matrix as an iterator of (vector, id) pairs
    def _matrix_iter(X):
        for i in range(X.shape[1]):
            yield X[:, i], str(i)

    results = run_gwas(Y, K, _matrix_iter(X), Kva=Kva, Kve=Kve, REML=True)

    print("Comparison of T-stats:", np.allclose(T_stats_old, results["F_STAT"], equal_nan=True))
    print("Comparison of P-values:", np.allclose(P_values_old, results["P_VALUE"], equal_nan=True))
