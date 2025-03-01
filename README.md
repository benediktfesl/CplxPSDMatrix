# Random Complex-Valued Hermitian (Covariance) Matrix Generator

This repository provides a simple Python function `make_cplx_spd_matrix` that generates a random complex-valued, Hermitian, and positive semi-definite (PSD) covariance matrix. 
This function is useful in scenarios involving complex-valued random variables, such as in signal processing and wireless communications.

## Overview

The function `make_cplx_spd_matrix` creates a random Hermitian matrix by following these steps:

1. Generates a random complex matrix $`A`$.
2. Computes the Singular Value Decomposition (SVD) of $`A^H A`$ (where $`A^H`$ is the conjugate transpose of $`A`$).
3. Constructs a new matrix $`C`$ using the SVD components and a random scaling matrix to ensure the resulting matrix is positive semi-definite.

The resulting matrix $`C`$ is Hermitian and PSD, suitable for use in simulations or applications requiring complex covariance matrices.

## Requirements

- Python 3.x
- `numpy` (for numerical operations)
- `scipy` (for linear algebra operations)

You can install the required packages with pip:

```bash
pip install numpy scipy
```

## Function Definition

```python
# Function to generate a random complex PSD matrix
def make_cplx_spd_matrix(dim, rng=np.random.default_rng(1234567)):
    # Generate a random complex matrix A
    A = rng.random([dim, dim]) + 1j * rng.random([dim, dim])

    # Perform Singular Value Decomposition (SVD) on A.T.conj() * A
    U, _, Vt = scipy.linalg.svd(np.dot(A.T.conj(), A), check_finite=False)

    # Construct the final matrix X using SVD components and random scaling
    X = np.dot(np.dot(U, 1.0 + np.diag(rng.random(dim))), Vt)

    # Return the complex PSD matrix
    return X
```

### Parameters:

- `dim`: The dimension of the square matrix (i.e., the number of rows and columns).
- `rng`: The random number generator. By default, it uses `np.random.default_rng(1234567)` for reproducibility.

### Returns:

- A complex-valued Hermitian positive semi-definite matrix of shape `(dim, dim)`.

## Related Function

There is a related function in the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html): `sklearn.datasets.make_spd_matrix`. However, this function only generates real-valued symmetric matrices, not complex-valued Hermitian ones.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
