import numpy as np
from mpi4py import MPI

"""
A collection of recycling schemes that we can bring into an SMC sampler.

"""


class ESS_Recycling:
    """

    Description
    -----------
    A reclying scheme that maximises the effective sample size, as
    described in Section 2.2 of https://arxiv.org/pdf/2004.12838.pdf

    Parameters
    ----------
    D : dimension of problem

    Author
    -------
    P.L.Green

    """

    def __init__(self, D):
        self.lr = np.array([])
        self.D = D

    def update_estimate(self, wn, k, mean_estimate, var_estimate):

        """
        Description
        -----------
        Esimates the mean and covariance matrix of the target, based on
        a recycling scheme that maximises the effective sample size.

        Parameters
        ----------
        wn : array of current normalised weights

        k : current iteration of the SMC sampler

        mean_estimate : (non-recycled) estimates of the target mean that
            have been realised so far

        var_estimate : (non-recycled) estimates of the target variance /
            covariance matrix that have been realised so far

        Returns
        -------
        mean_estimate_rc : recycled estimate of target mean  

        var_estimate_rc : recycled estimate of target variance /
            covariance matrix
        """

       
        # Find the new values of lr (equation (18) in
        # https://arxiv.org/pdf/2004.12838.pdf, though using normalised
        # weights)
        comm = MPI.COMM_WORLD
        wsqsum = np.zeros_like(1, dtype='d')
        comm.Allreduce(sendbuf=[np.sum(wn ** 2), MPI.DOUBLE], recvbuf=[wsqsum, MPI.DOUBLE], op=MPI.SUM)
        self.lr = np.append(self.lr, 1 / wsqsum)

        # Initialise c array (also defined by equation (18) in
        # https://arxiv.org/pdf/2004.12838.pdf)
        c = np.array([])

        # Loop to recalculate the optimal c values
        for k_dash in range(k + 1):
            c = np.append(c, self.lr[k_dash] / np.sum(self.lr))

        # Loop to recalculate estimates of the mean
        mean_estimate_rc = np.zeros([self.D])
        for k_dash in range(k + 1):
            mean_estimate_rc += c[k_dash] * mean_estimate[k_dash]

        # Loop to recalculate estimates of the variance / cov. matrix
        if self.D == 1:
            var_estimate_rc = np.zeros([self.D])
        else:
            var_estimate_rc = np.zeros([self.D, self.D])

        for k_dash in range(k + 1):

            # Define a 'correction' term, to account for the bias that
            # arises in the variance estimate as a result of using the
            # estimated mean
            correction = (mean_estimate_rc - mean_estimate[k_dash])**2

            # Variance estimate, including correction term
            var_estimate_rc += c[k_dash] * (var_estimate[k_dash] + correction)

        return mean_estimate_rc, var_estimate_rc
