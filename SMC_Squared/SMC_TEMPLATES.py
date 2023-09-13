from abc import abstractmethod, ABC

"""
Description
-----------
This module contains abstract methods for targets, initial proposals distributions 
and general proposal distributions
"""

class Target_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the target distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the target distribution, evaluated at x.

        """
        pass


class Q0_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the initial proposal distribution.

    """

    @abstractmethod
    def logpdf(self, x):
        """
        Description
        -----------
        Returns log pdf of the initial proposal, evaluated at x.
        """
        pass

    @abstractmethod
    def rvs(self, size):
        """
        Description
        -----------
        Returns samples from the initial proposal.

        Parameters
        ----------
        size : size of the sample being returned
        """
        pass

class Q_Base(ABC):
    """
    Description
    -----------
    This shows the methods that user will need to define to specify
    the general proposal distribution.
    """

    @abstractmethod
    def pdf(self, x, x_cond):
        """
        Description
        -----------
        Returns q(x | x_cond)
        """
        pass

    @abstractmethod
    def logpdf(self, x, x_cond):
        """
        Description
        -----------
        Returns log q(x | x_cond)
        """
        pass

    @abstractmethod
    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a single sample from the proposal, q(x | x_cond).
        """

        pass
