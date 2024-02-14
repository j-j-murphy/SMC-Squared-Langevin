import autograd.numpy as np
import importance_sampling as IS
from SMC_TEMPLATES import Q_Base
from RECYCLING import ESS_Recycling
from RNG import RNG
from mpi4py import MPI
from resampling import systematic_resampling
from normalisation import normalise
import copy
from SMC_DIAGNOSTICS import smc_no_diagnostics
#from STAN_MODEL import StanModel


class SMC():

    """
    Description
    -----------
    A base class for an SMC sampler. Estimates of the mean and variance
    / covariance matrix associated with a specific iteration are reported
    in mean_estimate and var_estimate respectively. Recycled estimates of
    the mean and variance / covariance matrix are reported in 
    mean_estimate_rc and var_estimate_rc respectively (when recycling is
    active). 
    Parameters
    ----------
    N : no. of samples generated at each iteration
    D : dimension of the target distribution
    p : target distribution instance
    q0 : initial proposal instance
    K : no. iterations to run
    q : general proposal distribution instance
    optL : approximation method for the optimal L-kernel. Can be either
        'gauss' or 'monte-carlo' (representing a Gaussian approximation
        or a Monte-Carlo approximation respectively).
    rc_scheme : option to have various recycling schemes (or none) Can 
        currently be 'ESS_Recycling' (which aims to maximise the effective
        sample size) or 'None' (which is currently the default). 
    verbose : option to print various things while the sampler is running
    Methods
    -------
    generate_samples : runs the SMC sampler to generate weighted
        samples from the target.
    update_weights : updates the log weight associated with each sample
        i.e. evaluates the incremental weights.
    Author
    ------
    P.L.Green and L.J. Devlin
    """

    def __init__(self, N, D, p, q0, K, proposal, optL, seed, prop, 
                 rc_scheme=None, verbose=False, diagnose=None):

        # Assign variables to self
        self.N = N
        self.loc_n = int(N/MPI.COMM_WORLD.Get_size())
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.optL = optL
        self.seed = seed
        self.prop = prop
        self.verbose = verbose
        self.rank = MPI.COMM_WORLD.Get_rank()

        # Can either have a user-defined proposal or random walk
        # proposal in this implementation
        if(isinstance(proposal, Q_Base)):
            self.q = proposal
            self.proposal='user'
        elif(proposal == 'rw'):
            from proposals.random_walk import random_walk_proposal
            self.q = random_walk_proposal(self.D)
            self.proposal = 'rw'

        # Initialise recycling scheme. For now we just have one,
        # but we might add some more later!
        if rc_scheme == 'ESS_Recycling':
            self.rc = ESS_Recycling(self.D)
        elif rc_scheme == None:
            self.rc = None

        if diagnose == None:
            self.diagnose = smc_no_diagnostics()
        else:
            self.diagnose = diagnose

    def generate_samples(self):
	
        seed = self.seed
        mvrs_rng = RNG(seed)
        #rngs = [RNG(i + seed) for i in range(self.loc_n)]
        rngs = [RNG(i + self.rank*self.loc_n + 1 + seed) for i in range(self.loc_n)]
        """
        Description
        -----------
        Run SMC sampler to generate weighted samples from the target.
        """

        # Initialise arrays for storing samples (x_new)
        x_new = np.zeros([self.loc_n, self.D])
        x = np.zeros([self.loc_n, self.D])

        # Runtimes for each iteration
        self.runtimes = np.zeros(self.K)
        start = MPI.Wtime()

        # Initilise estimates of target mean and covariance matrix,
        # where 'rc' represents the overall estimate (i.e. after
        # recyling). 
        self.mean_estimate = np.zeros([self.K, self.D])
        self.mean_estimate_rc = np.zeros([self.K, self.D])
        if self.D == 1:
            self.var_estimate = np.zeros([self.K, self.D])
            if self.rc:
                self.var_estimate_rc = np.zeros([self.K, self.D])
        else:
            self.var_estimate = np.zeros([self.K, self.D, self.D])
            if self.rc:
                self.var_estimate_rc = np.zeros([self.K, self.D, self.D])

        # Used to record the effective sample size and the points
        # where resampling occurred.
        self.Neff = np.zeros(self.K)
        self.resampling_points = np.array([])

        # Sample from prior and find initial evaluations of the
        # target and the prior. Note that, be default, we keep
        # the log weights vertically stacked.        
        for ii in range(self.loc_n):        
            x[ii] = self.q0.rvs(1, rngs[ii])
            
        p_logpdf_x =np.zeros(self.loc_n)
        p_logpdf_x_grads =np.zeros([self.loc_n, self.D])
        p_logpdf_x_grads_2 =np.zeros([self.loc_n, self.D, self.D])
        p_log_q0_x =np.zeros(self.loc_n)
        
        for ii in range(self.loc_n):
            p_logpdf_x[ii], p_logpdf_x_grads[ii], p_logpdf_x_grads_2[ii] = self.p.logpdf(x[ii], rngs[ii])
            p_log_q0_x[ii] = self.q0.logpdf(x[ii])
        p_logpdf_x  = np.vstack(p_logpdf_x)
        p_log_q0_x = np.vstack(p_log_q0_x)
        
        #log = np.zeros(self.loc_n)
        # Find log-weights of prior samples
        logw = p_logpdf_x - p_log_q0_x    
        
        # Main sampling loop
        for self.k in range(self.K):

            if self.verbose and self.rank==1:
                print('\nIteration :', self.k)

            # Find normalised weights and realise estimates
            wn = np.exp(normalise(logw))
            
            #print(x)
            (self.mean_estimate[self.k],
             self.var_estimate[self.k]) = IS.estimate(x, wn, self.D, self.N)
             
            self.diagnose.save_iter_info(self.k, x, x, logw, self.seed)

            # Recycling scheme
            if self.rc:
                (self.mean_estimate_rc[self.k], 
                 self.var_estimate_rc[self.k]) = self.rc.update_estimate(wn, self.k, 
                                                                         self.mean_estimate, 
                                                                         self.var_estimate)


            # Record effective sample size at kth iteration
            self.Neff[self.k] = IS.ess(wn)

            # Resample if effective sample size is below threshold
            if self.Neff[self.k] < self.N/2:
                print("Resampling")
                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)								   
                x, p_logpdf_x, wn, p_logpdf_x_grads, p_logpdf_x_grads_2 = systematic_resampling(x, p_logpdf_x, wn, self.N, mvrs_rng, p_logpdf_x_grads, p_logpdf_x_grads_2)
                logw = np.log(wn)
                print("p_logpdf_x_grads_2", p_logpdf_x_grads_2)
            # Propose new samples
            for i in range(self.loc_n):
                print("proposal hessian", p_logpdf_x_grads_2[i])
                x_new[i] = self.q.rvs(x_cond=x[i], rngs=rngs[i], grads=p_logpdf_x_grads[i], grads_1=p_logpdf_x_grads_2[i], props=self.prop)

            print("x", x)
            print("x_new", x_new)
	        
            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new =np.zeros(self.loc_n)
            p_logpdf_x_grads_new = np.zeros((self.loc_n, self.D))
            p_logpdf_x_grads_new_1 = np.zeros((self.loc_n, self.D, self.D))
            
            for ii in range(self.loc_n):
                p_logpdf_x_new[ii], p_logpdf_x_grads_new[ii], p_logpdf_x_grads_new_1[ii] = self.p.logpdf(x_new[ii], rngs[ii])
           
            p_logpdf_x_new  = np.vstack(p_logpdf_x_new)

            # print("p_logpdf_x_grads_new", p_logpdf_x_grads_new)
            # p_logpdf_x_grads_new  = np.vstack(p_logpdf_x_grads_new)
            # print("p_logpdf_x_grads_new after", p_logpdf_x_grads_new)

            # print("p_logpdf_x_grads_new_1", p_logpdf_x_grads_new_1)
            # p_logpdf_x_grads_new_1  = np.vstack(p_logpdf_x_grads_new_1)
            # print("p_logpdf_x_grads_new_1 after", p_logpdf_x_grads_new_1)

            # Update log weights
            logw_new = self.update_weights(x, x_new, logw, p_logpdf_x,
                                           p_logpdf_x_new)
                                           
            # Make sure that, if p.logpdf(x_new) is -inf, then logw_new
            # will also be -inf. Otherwise it is returned as NaN.
            for i in range(self.loc_n):
                if p_logpdf_x_new[i] == -np.inf:
                    logw_new[i] = -np.inf
                elif logw[i] == -np.inf:
                    logw_new[i] = -np.inf

            # Update samples, log weights, and posterior evaluations
            x = np.copy(x_new)
            logw = np.copy(logw_new)
            p_logpdf_x = np.copy(p_logpdf_x_new)
            p_logpdf_x_grads = np.copy(p_logpdf_x_grads_new)
            p_logpdf_x_grads_2 = np.copy(p_logpdf_x_grads_new_1)
            self.runtimes[self.k] = MPI.Wtime() - start

        # Final quantities to be assigned to self
        self.x = x
        self.logw = logw
        if self.rank==0:
        	self.diagnose.save_final_info(self.mean_estimate, self.mean_estimate_rc, 
                                        self.var_estimate, self.var_estimate_rc, 
                                        self.Neff, self.resampling_points, self.N,
                                        self.runtimes)
    
    def update_weights(self, x, x_new, logw, p_logpdf_x,
                       p_logpdf_x_new):
        """
        Description
        -----------
        Used to update the log weights of a new set of samples, using the
            weights of the samples from the previous iteration. This is
            either done using a Gaussian approximation or a Monte-Carlo
            approximation of the opimal L-kernel.

        Parameters
        ----------
        x : samples from the previous iteration

        x_new : samples from the current iteration

        logw : low importance weights associated with x

        p_logpdf_x : log target evaluations associated with x

        p_logpdf_x_new : log target evaluations associated with x_new

        Returns
        -------
        logw_new : log weights associated with x_new

        """

        # Initialise
        logw_new = np.vstack(np.zeros(self.loc_n))

        # Use the forwards proposal as the L-kernel
        if self.optL == 'forwards-proposal':
            # Find new weights
            for i in range(self.loc_n):  # Changed N to loc_n
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i])

        # Use Gaussian approximation of the optimal L-kernel
        # (see Section 4.1 of https://www.sciencedirect.com/science/article/pii/S0888327021004222
        #  or Section 4.1 of https://arxiv.org/pdf/2004.12838.pdf)
        if self.optL == 'gauss':

            # Collect x and x_new together into X
            X = np.hstack([x, x_new])

            # Directly estimate the mean and covariance matrix of X
            mu_X = IS.weighted_mean(x=X, wn=np.repeat(1/self.N, self.loc_n))  # mu_X = np.mean(X, axis=0)
            cov_X = IS.covar(wn=np.repeat(1/self.N, self.loc_n), x=X, m=mu_X)  # np.cov(np.transpose(X))

            # Find mean of the joint distribution (p(x, x_new))
            mu_x, mu_xnew = mu_X[0:self.D], mu_X[self.D:2 * self.D]

            # Find covariance matrix of joint distribution (p(x, x_new))
            (cov_x_x,
             cov_x_xnew,
             cov_xnew_x,
             cov_xnew_xnew) = (cov_X[0:self.D, 0:self.D],
                               cov_X[0:self.D, self.D:2 * self.D],
                               cov_X[self.D:2 * self.D, 0:self.D],
                               cov_X[self.D:2 * self.D, self.D:2 * self.D])

            # Define new L-kernel
            def L_logpdf(x, x_cond):

                # Mean of approximately optimal L-kernel
                mu = (mu_x + cov_x_xnew @ np.linalg.inv(cov_xnew_xnew) @
                      (x_cond - mu_xnew))

                # Variance of approximately optimal L-kernel
                cov = (cov_x_x - cov_x_xnew @
                       np.linalg.inv(cov_xnew_xnew) @ cov_xnew_x)

                # Add ridge to avoid singularities
                cov += np.eye(self.D) * 1e-6

                # Log det covariance matrix
                sign, logdet = np.linalg.slogdet(cov)
                log_det_cov = sign * logdet

                # Inverse covariance matrix
                inv_cov = np.linalg.inv(cov)

                # Find log pdf
                logpdf = (-0.5 * log_det_cov -
                          0.5 * (x - mu).T @ inv_cov @ (x - mu))

                return logpdf

            # Find new weights
            for i in range(self.loc_n):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                               L_logpdf(x[i], x_new[i]) -
                               self.q.logpdf(x_new[i], x[i]))

        # Use Monte-Carlo approximation of the optimal L-kernel
        # (not published at the moment but see
        # https://www.overleaf.com/project/6130ff176124735112a885b3)
        if self.optL == 'monte-carlo':
            for i in range(self.N):

                # Initialise what will be the denominator of our
                # weight-update equation
                den = np.zeros(1)

                # Realise Monte-Carlo estimate of denominator
                for j in range(self.N):
                    den += self.q.pdf(x_new[i], x[j])
                den /= self.N

                # Calculate new log-weight
                logw_new[i] = p_logpdf_x_new[i] - np.log(den)

        return logw_new
