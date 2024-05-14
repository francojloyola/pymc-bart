# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
from numpy cimport (
    ndarray,
    float64_t,
    int64_t,
)
from cython import boundscheck, wraparound
from cython cimport Py_ssize_t
cimport cython

DTYPE = np.float64
ctypedef float64_t DTYPE_t
ITYPE = np.int64
ctypedef int64_t ITYPE_t

def fast_mean(ndarray ari):
    """Use Cyython to speed up the computation of the mean."""
    if ari.ndim == 1:
        return fast_mean_ndim1(ari)
    else:
        return fast_mean_ndim2(ari)



def fast_mean_ndim1(DTYPE_t[:] ari):
    cdef Py_ssize_t count = ari.shape[0]
    cdef double suma = 0
    for i in range(count):
        suma += ari[i]
    return suma / count


def fast_mean_ndim2(DTYPE_t[:,:] ari):
    cdef Py_ssize_t rows = ari.shape[0]
    cdef Py_ssize_t cols = ari.shape[1]
    cdef  ndarray[DTYPE_t,ndim=1] result = np.zeros(rows, dtype=DTYPE)

    for i in range(rows):
        total = 0.0
        for j in range(cols):
            total += ari[i, j]
        result[i] = total / cols

    return result


def fast_linear_fit(
        ndarray[DTYPE_t] x,
        ndarray[DTYPE_t] y,
        int m,
        ndarray[DTYPE_t] norm
):
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t r = y.shape[0]
    cdef ndarray[DTYPE_t] y_fit = np.empty_like(y)
    cdef ndarray[DTYPE_t] xbar = np.sum(x) / n
    cdef ndarray[DTYPE_t] ybar = np.sum(y, axis=1) / n
    cdef ndarray[DTYPE_t] x_diff = x - xbar
    cdef ndarray[DTYPE_t] y_diff = y - np.expand_dims(ybar, axis=1)
    cdef ndarray[DTYPE_t] x_var = np.dot(x_diff, x_diff.T)
    cdef ndarray[DTYPE_t] b
    cdef ndarray[DTYPE_t] a

    if x_var == 0:
        b = np.zeros(r)
    else:
        b = np.dot(x_diff, y_diff.T) / x_var

    a = ybar - b * xbar

    for i in range(r):
        y_fit[i, :] = a[i] + b[i] * x

    return y_fit, [a, b]

# @njit
# def fast_linear_fit(
#     x: npt.NDArray[np.float_],
#     y: npt.NDArray[np.float_],
#     m: int,
#     norm: npt.NDArray[np.float_],
# ) -> Tuple[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]:
#     n = len(x)
#     y = y / m + np.expand_dims(norm, axis=1)
#
#     xbar = np.sum(x) / n
#     ybar = np.sum(y, axis=1) / n
#
#     x_diff = x - xbar
#     y_diff = y - np.expand_dims(ybar, axis=1)
#
#     x_var = np.dot(x_diff, x_diff.T)
#
#     if x_var == 0:
#         b = np.zeros(y.shape[0])
#     else:
#         b = np.dot(x_diff, y_diff.T) / x_var
#
#     a = ybar - b * xbar
#
#     y_fit = np.expand_dims(a, axis=1) + np.expand_dims(b, axis=1) * x
#     return y_fit.T, [a, b]


cpdef ndarray[ITYPE_t] inverse_cdf(ndarray[DTYPE_t, ndim=1] single_uniform,
                                      ndarray[DTYPE_t, ndim=1] normalized_weights):
    cdef Py_ssize_t idx = 0
    cdef double a_weight = normalized_weights[0]
    cdef Py_ssize_t sul = single_uniform.shape[0]
    cdef ndarray[ITYPE_t, ndim=1] new_indices = np.empty(sul, dtype=ITYPE)
    cdef Py_ssize_t i

    for i in range(sul):
        while single_uniform[i] > a_weight:
            idx += 1
            a_weight += normalized_weights[idx]
        new_indices[i] = idx

    return new_indices


def get_split_data_points(ITYPE_t[:] idx_data_points,to_left):
    idx_dp = np.asarray(idx_data_points)
    return idx_dp[to_left], idx_dp[~to_left]