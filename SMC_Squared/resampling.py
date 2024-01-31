from mpi4py import MPI
import numpy as np
from mpi4py import MPI
from prefix_sum import inclusive_prefix_sum
from fixed_size_redistribution.redistribution import centralised_redistribution, redistribute


def get_number_of_copies(wn, rng):
    comm = MPI.COMM_WORLD
    N = len(wn) * comm.Get_size()

    cdf = inclusive_prefix_sum(wn*N)  # np.exp(wn)*N if the weights are in log scale
    cdf_of_i_minus_one = cdf - np.reshape(wn*N, newshape=cdf.shape)  # np.exp(wn)*N if the weights are in log scale

    u = np.array(rng.randomUniform(0.0, 1.0), dtype=wn.dtype)
    comm.Bcast(buf=[u, MPI._typedict[u.dtype.char]], root=0)

    ncopies = np.ceil(cdf - u) - np.ceil(cdf_of_i_minus_one - u)
    return ncopies.astype('int32')


def systematic_resampling(x, p_logpdf_x, wn, N, rng, grads, grads_1):
    np.random.seed(42)
    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    loc_n = int(N / P)
    dim = x.shape[1]
    #need to flatten grads_1(hessian)
    
    grads = grads.reshape(-1, 1)
    grads_1 = grads_1.reshape(-1, 1)
    x = np.concatenate((x, grads, grads_1), axis=1)
    
    ncopies = get_number_of_copies(wn, rng)
    p_logpdf_x_shape = p_logpdf_x.shape
    x = np.hstack((x, np.reshape(p_logpdf_x, newshape=(len(p_logpdf_x), 1))))
    x = redistribute(x, ncopies)
    
    #unpack x, grads, hessian
    x = x[:, 0:x.shape[1]]
    grads_new = x[:, dim:(2*dim)].reshape(len(grads),)
    grads_new_1 = x[:, (2*dim):(2*dim+dim**2)].reshape(len(grads_new),)

    wn_new = np.ones(loc_n) / N
    x_new = x[:, 0:dim]
    p_logpdf_x_new = np.reshape(x[:, -1], newshape=p_logpdf_x_shape)

    return x_new, p_logpdf_x_new, wn_new, grads_new, grads_new_1