from mpi4py import MPI
import numpy as np
from mpi4py import MPI
from smc_components.prefix_sum import inclusive_prefix_sum
from smc_components.fixed_size_redistribution.redistribution import centralised_redistribution, redistribute


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

    # print("Before resampling_______")
    # print("x", x)
    # print("p_logpdf_x", p_logpdf_x)
    # print("wn", wn)
    # print("grads", grads)
    # print("grads_1", grads_1)
    # print("_________________________")

    grads = grads
    grads_1 = grads_1.reshape(-1, dim**2)
    p_logpdf_x = p_logpdf_x.reshape(-1, 1)
    x = np.concatenate((x, grads, grads_1, p_logpdf_x), axis=1)
    
    ncopies = get_number_of_copies(wn, rng)
    x = redistribute(x, ncopies)
    
    x_new = x[:, 0:dim]
    grads_new = x[:, dim:(2*dim)]
    grads_new_1 = x[:, (2*dim):(2*dim+dim**2)].reshape(len(grads_new), dim, dim)
    p_logpdf_x_new = x[:, -1].reshape(len(p_logpdf_x), )

    # print("After resampling_______")
    # print("x_new", x_new)
    # print("p_logpdf_x_new", p_logpdf_x_new)
    # print("grads_new", grads_new)
    # print("grads_new_1", grads_new_1)
    # print("_________________________")
    wn_new = np.ones(loc_n) / N    

    return x_new, p_logpdf_x_new, wn_new, grads_new, grads_new_1