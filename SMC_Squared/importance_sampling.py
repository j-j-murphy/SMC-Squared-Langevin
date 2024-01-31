import numpy as np
from mpi4py import MPI

"""
A collection of methods that relate to importance sampling.

P.L.Green
"""


def normalise_weights(logw):
    """
    Description
    -----------
    Normalise importance weights. Note that we remove the mean here
        just to avoid numerical errors in evaluating the exponential.
        We have to be careful with -inf values in the log weights
        sometimes. This can happen if we are sampling from a pdf with
        zero probability regions, for example.

    Parameters
    ----------
    logw : array of logged importance weights

    Returns
    -------
    wn : array of normalised weights

    """
    comm = MPI.COMM_WORLD

    # Identify elements of logw that are not -inf
    logw_copy = np.copy(logw)
    logw_copy[np.isnan(logw)] = -np.inf
    indices = np.invert(np.isneginf(logw_copy))
    #print(indices)

    # Apply normalisation only to those elements of log that are not -inf
    wmax = np.zeros_like(1, dtype='d')
    comm.Allreduce(sendbuf=[np.max(logw_copy[indices]), MPI.DOUBLE], recvbuf=[wmax, MPI.DOUBLE], op=MPI.MAX)
    logw_copy[indices] = logw_copy[indices] - wmax

    # Find standard weights
    w = np.exp(logw_copy)
    
    # Find normalised weights
    wsum = np.zeros_like(1, dtype='d')
    comm.Allreduce(sendbuf=[np.sum(w), MPI.DOUBLE], recvbuf=[wsum, MPI.DOUBLE], op=MPI.SUM)
    wn = w / wsum

    return wn

def ess(wn):
    """
    Description
    -----------
    Computes the Effective Sample Size of the given normalised weights

    Parameters
    ----------
    wn : array of normalised weights

    Returns
    -------
    double scalar : Effective Sample Size

    """
    comm = MPI.COMM_WORLD
    glob_sum = np.zeros_like(1, dtype='d')  # global sum
    loc_sum = np.sum(np.square(wn))
    comm.Allreduce(sendbuf=[loc_sum, MPI.DOUBLE], recvbuf=[glob_sum, MPI.DOUBLE], op=MPI.SUM)

    return 1 / glob_sum


def resample(x, p_logpdf_x, wn, N, rng, grads, grads_1):
    """
    Description
    -----------
    Resample given normalised weights.

    Parameters
    ----------
    x : array of current samples

    p_logpdf_x : array of current target evaluations.

    wn : array or normalised weights

    N : no. samples

    Returns
    -------
    x_new : resampled values of x

    p_logpdf_x_new : log pdfs associated with x_new

    wn_new : normalised weights associated with x_new

    """
    #np.random.seed(seed=rngs)
    i = np.linspace(0, N-1, N, dtype=int)  # Sample positions
    i_new = rng.randomChoice(i, N, wn[:, 0])   # i is resampled
    wn_new = np.ones(N) / N           # wn is reset

    # New samples
    x_new = x[i_new]
    grads_new = grads[i_new]
    grads_new_1 = grads_1[i_new]
    p_logpdf_x_new = p_logpdf_x[i_new]

    return x_new, p_logpdf_x_new, wn_new, grads_new, grads_new_1
    
    
def weighted_mean(x, wn):
    
    comm = MPI.COMM_WORLD
    loc_mean = wn.T @ x
    mean = np.zeros_like(loc_mean)
    
    comm.Allreduce(loc_mean, mean, op=MPI.SUM)
    
    return mean


def covar(wn, x, m):
    
    comm = MPI.COMM_WORLD
    loc_n, D = x.shape[0], x.shape[1]
    x = x - m
    
    if D ==1:
        loc_v = wn.T @ np.square(x)
        v = np.zeros_like(loc_v)
    
    else:
        loc_v = np.zeros([D, D])
        v = np.zeros_like(loc_v)
        for i in range(loc_n):
            
            xv = x[i][np.newaxis]
            loc_v += wn[i] * xv.T @ xv
     
    comm.Allreduce(loc_v, v, op=MPI.SUM)
    return v
    
    


def estimate(x, wn, D, N):
    """
    Description
    -----------
    Estimate some quantities of interest (just mean and covariance
        matrix for now).

    Parameters
    ----------
    x : samples from the target

    wn : normalised weights associated with the target

    D : dimension of problem

    N : no. samples

    Returns
    -------
    m : estimated mean

    v : estimated covariance matrix

    """

    # Estimate the mean
    m = weighted_mean(x=x, wn=wn)           #wn.T @ x
    v = covar(wn=wn, x=x, m=m)
    

    # Remove the mean from our samples then estimate the variance
   # x = x - m

   # if D == 1:
   #     v = wn.T @ np.square(x)
   # else:
   #     v = np.zeros([D, D])
   #     for i in range(N):
   #         xv = x[i][np.newaxis]  # Make each x into a 2D array
   #         v += wn[i] * xv.T @ xv

    return m, v
