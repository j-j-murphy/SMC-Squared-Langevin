from SMC_BASE import Q_Base
import autograd.numpy as np
from autograd import elementwise_grad as egrad
from scipy.stats import multivariate_normal

class HMC_proposal(Q_Base):

    """
        Description
        -----------
        Fixed step Hamiltonian Monte Carlo proposal distribution for an SMC-sampler. 
        Moves samples around the target using the leapfrog integration method over a fixed
        number of steps and a fixed step-size. In HMC omentum is usually used in Hamiltonian
        MCMC, but here we assume (for the time being) that the mass is the identity matrix,
        we therefore refer to it as velocity since momentum = mass * velocity
     
        
        Parameters
        ----------
        D: Dimension of target
        
        p : target distribution

        h : Step size used by Leapfrog

        Steps : no. of steps made by Leapfrog

        Cov: Scalar term which increases the variance of the velocity distribution
          
        Author
        ------
        L.J. Devlin

    
    """

    def __init__(self, D, p, h, steps, Cov):
        self.D = D
        self.target = p
        self.h=h
        self.steps=steps
        
        # Set a gradient object which we call each time we require it inside Leapfrog
        self.grad=egrad(self.target.logpdf)

        # Define an initial velocity disitrbution
        self.v_dist = multivariate_normal(mean=np.zeros(D), cov=Cov*np.eye(D))

        
    def pdf(self, v, v_cond = None):
        """
        Description
        -----------
        Calculate pdf of velocity distribution
        """

        return self.v_dist.pdf(v)

    def logpdf(self, v, v_cond = None):
        """
        Description
        -----------
        Calculate logpdf of velocity distribution
        """

        return self.v_dist.logpdf(v)

    def rvs(self, x_cond):
        """
        Description
        -----------
        Returns a new sample state at the end of the integer number of Leapfrog steps.
        """

        # Unpack position, initial velocity, and initial gradient 
        x = x_cond[0,:] 
        v = x_cond[1,:]
        grad_x=x_cond[2, :]

        x_new, v_new = self.generate_HMC_samples(x, v, grad_x)
        return x_new, v_new, self.h*self.steps


    def generate_HMC_samples(self, x, v, grad_x):
        
        """
        Description
        -----------
        Handles the fixed step HMC proposal by generating a new sample after a number of Leapfrog steps.
        """

        # Main leapfrog loop
        for k in range(0,self.steps):
            x, v, grad_x = self.Leapfrog(x, v, grad_x)
        
        return x, v


    def Leapfrog(self, x, v, grad_x):
        
        """
        Description
        -----------
        Performs a single Leapfrog step returning the final position, velocity and gradient.
        """
        
        v = np.add(v, (self.h/2)*grad_x)
        x = np.add(x, self.h*v)
        grad_x = self.grad(x)
        v = np.add(v, (self.h/2)*grad_x)
        
        return x, v, grad_x


    def v_rvs(self, size):

        """
        Description
        -----------
        Draw a number of samples equal to size from the velocity/momentum distribution
        """
        
        return self.v_dist.rvs(size)