from typing import Callable, List, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg


__all__ = ['diffusion_maps']


KernelMatrix = sparse.coo_matrix
PPrime = sparse.csr_matrix


def _get_kernel_matrix(data: np.ndarray,
                       kernel: Callable[[np.ndarray, np.ndarray], float],
                       tol: float) -> KernelMatrix:
    n_data: int = data.shape[0]

    kernel_data: List[float] = []
    kernel_row_ind: List[int] = []
    kernel_col_ind: List[int] = []

    def kernel_append(row_ind: int, col_ind: int, value: float) -> None:
        kernel_data.append(value)
        kernel_row_ind.append(row_ind)
        kernel_col_ind.append(col_ind)

    for i in range(n_data):
        for j in range(i + 1):
            value = kernel(data[i], data[j])

            if value >= tol:
                kernel_append(i, j, value)
                if i != j:
                    kernel_append(j, i, value)

    return sparse.coo_matrix((kernel_data, (kernel_row_ind, kernel_col_ind)),
                             shape=(n_data, n_data))


def _get_Pprime(kernel_matrix: KernelMatrix, d_nh: np.ndarray) -> PPrime:
    n_data: int = kernel_matrix.shape[0]

    Pprime_data: np.ndarray = np.empty_like(kernel_matrix.data)

    for i in range(len(kernel_matrix.data)):
        Pprime_data[i] = kernel_matrix.data[i] * \
            d_nh[kernel_matrix.row[i]] * d_nh[kernel_matrix.col[i]]

    return sparse.csr_matrix((Pprime_data,
                              (kernel_matrix.row, kernel_matrix.col)),
                             shape=(n_data, n_data))


def _eigendecompose_transition_matrix(
        Pprime: PPrime, d_nh: np.ndarray, n_eigen: int,
        tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, v = slinalg.eigsh(Pprime, k=n_eigen, which='LM', tol=tol)

    assert abs(w[-1] - 1) <= max(tol, np.finfo(Pprime.dtype).eps), \
        "maximum eigenvalue of transition matrix is not 1"

    right_eigenvectors = v * d_nh[:, np.newaxis]
    left_eigenvectors = v / d_nh[:, np.newaxis]

    return (w, right_eigenvectors, left_eigenvectors)


def diffusion_maps(
        data: np.ndarray, kernel: Callable[[np.ndarray, np.ndarray], float],
        out_dims: int, *, kernel_tol: float = 5e-6,
        eig_tol: float = 5e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_matrix = _get_kernel_matrix(data, kernel, kernel_tol)

    d = np.reshape(np.array(kernel_matrix.sum(axis=1)), -1)
    d_nh = d ** -0.5

    Pprime = _get_Pprime(kernel_matrix, d_nh)

    eigenvalues, left_eigenvectors, right_eigenvectors \
        = _eigendecompose_transition_matrix(Pprime, d_nh, out_dims, eig_tol)

    return eigenvalues, left_eigenvectors, right_eigenvectors
