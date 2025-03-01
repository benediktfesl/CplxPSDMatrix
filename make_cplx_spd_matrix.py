import numpy as np
import scipy

def make_cplx_spd_matrix(dim, rng=np.random.default_rng(1234567)):
    A = rng.random([dim, dim]) + 1j*rng.random([dim, dim])
    U, _, Vt = scipy.linalg.svd(np.dot(A.T.conj(), A), check_finite=False)
    X = np.dot(np.dot(U, 1.0 + np.diag(rng.random(n_dim))), Vt)
    return X
