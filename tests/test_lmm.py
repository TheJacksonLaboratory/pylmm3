import numpy as np
from pylmm.lmm import LMM  # Make sure the LMM class is compatible with Python 3

# I generated this data by using the python 2 pylmm library.
prefix=".local/lmm/"

# Load the data
Y = np.load(prefix+'Y.npy')
X0 = np.load(prefix+'X0.npy')
K = np.load(prefix+'K.npy')
Kva = np.load(prefix+'Kva.npy')
Kve = np.load(prefix+'Kve.npy')
hmax_old = np.load(prefix+'hmax.npy')
beta_old = np.load(prefix+'beta.npy')
sigma_old = np.load(prefix+'sigma.npy')
LL_old = np.load(prefix+'LL.npy')

# Re-initialize and fit the LMM model
lmm = LMM(Y, K, Kva, Kve, X0)
hmax_new, beta_new, sigma_new, LL_new = lmm.fit()

# Compare the results
print("Comparison of hmax:", np.allclose(hmax_old, hmax_new))
print("Comparison of beta:", np.allclose(beta_old, beta_new))
print("Comparison of sigma:", np.allclose(sigma_old, sigma_new))
print("Comparison of log-likelihood:", np.allclose(LL_old, LL_new))
