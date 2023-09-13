from SMC_BASE import Q_Base
import numpy as np


class random_walk_proposal(Q_Base):

    """
        Description
        -----------
        Generic Random Walk proposal distribution with unit variance.

    """

    def __init__(self, D):
        self.D = D

    def pdf(self, x, x_cond):
        """
        Description
        -----------
        Returns pdf from a normal distribution (with unit variance and
        mean x_cond) for parameter x.
        """

        dx = np.vstack(x - x_cond)
        p = (2*np.pi)**(-self.D/2) * np.exp(-0.5 * dx.T @ dx)

        return p[0]

    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns logpdf from a normal distribution (with unit variance and
        mean x_cond) for parameter x.
        """

        dx = np.vstack(x - x_cond)
        logp = -self.D/2 * np.log(2*np.pi) - 0.5 * dx.T @ dx
        return logp

    def rvs(self, x_cond, rngs):
        """
        Description
        -----------
        Returns a new sample state based on a standard normal Gaussian
        random walk.
        """
        #np.random.seed(seed=rngs)
        return x_cond + 0.1 * np.random.randn(self.D)
