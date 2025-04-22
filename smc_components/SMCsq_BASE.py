import autograd.numpy as np
import smc_components.importance_sampling as IS
from smc_components.SMC_TEMPLATES import Q_Base
from smc_components.RECYCLING import ESS_Recycling
from smc_components.RNG import RNG
from mpi4py import MPI
from smc_components.resampling import systematic_resampling
from smc_components.normalisation import normalise
import copy
from smc_components.SMC_DIAGNOSTICS import smc_no_diagnostics

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

    def __init__(self, N, D, p, q0, K, proposal, optL, seed, 
                 rc_scheme=None, verbose=False, diagnose=None):
        self.N = N
        self.loc_n = int(N/MPI.COMM_WORLD.Get_size())
        self.D = D
        self.p = p
        self.q0 = q0
        self.K = K
        self.optL = optL
        self.seed = seed
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
        rngs = [RNG(i + self.rank*self.loc_n + 1 + seed) for i in range(self.loc_n)]

        # Initialise arrays for storing samples (x_new)
        x_new = np.zeros([self.loc_n, self.D])
        x = np.zeros([self.loc_n, self.D])
        v = np.zeros([self.loc_n, self.D])
        v_half = np.zeros([self.loc_n, self.D])

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
            x[ii] = self.q0.rvs(rngs[ii])

        p_logpdf_x =np.zeros(self.loc_n)
        p_logpdf_x_grads =np.zeros([self.loc_n, self.D])
        p_logpdf_x_hess =np.zeros([self.loc_n, self.D, self.D])
        p_log_q0_x =np.zeros(self.loc_n)
        
        for ii in range(self.loc_n):
            p_logpdf_x[ii], p_logpdf_x_grads[ii], p_logpdf_x_hess[ii] = self.p.logpdf(x[ii], rngs[ii])
            p_log_q0_x[ii] = self.q0.logpdf(x[ii])
            
        p_logpdf_x  = np.vstack(p_logpdf_x)
        p_log_q0_x = np.vstack(p_log_q0_x)
        
        # Find log-weights of prior samples
        logw = p_logpdf_x - p_log_q0_x    
        # Main sampling loop
        for self.k in range(self.K):

            if self.verbose and self.rank==0:
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
                if self.verbose and self.rank==0:
                    print("Resampling")
                self.resampling_points = np.append(self.resampling_points,
                                                   self.k)								   
                x, p_logpdf_x, wn, p_logpdf_x_grads, p_logpdf_x_hess = systematic_resampling(x, p_logpdf_x, wn, self.N, mvrs_rng, p_logpdf_x_grads, p_logpdf_x_hess)
                logw = np.log(wn)

            # Propose new samples
            for i in range(self.loc_n):
                x_new[i], v[i], v_half[i] = self.q.rvs(x_cond=x[i], rngs=rngs[i], grads=p_logpdf_x_grads[i], hess=p_logpdf_x_hess[i])
	        
            # Make sure evaluations of likelihood are vectorised
            p_logpdf_x_new =np.zeros(self.loc_n)
            p_logpdf_x_grads_new = np.zeros((self.loc_n, self.D))
            p_logpdf_x_hess_new = np.zeros((self.loc_n, self.D, self.D))
            
            for ii in range(self.loc_n):
                p_logpdf_x_new[ii], p_logpdf_x_grads_new[ii], p_logpdf_x_hess_new[ii] = self.p.logpdf(x_new[ii], rngs[ii])
            
            if self.k%5==0 and self.rank==0:
                print("k", self.k)
                print("mean estimate", self.mean_estimate[self.k])
                print("grads", p_logpdf_x_grads_new)
                print("second order grads", p_logpdf_x_hess_new)
            p_logpdf_x_new  = np.vstack(p_logpdf_x_new)

            # Update log weights
            logw_new = self.update_weights(x, x_new, v, v_half, logw, p_logpdf_x,
                                           p_logpdf_x_new, p_logpdf_x_grads, p_logpdf_x_grads_new, p_logpdf_x_hess, 
                                           p_logpdf_x_hess_new)
                                           
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
            p_logpdf_x_hess = np.copy(p_logpdf_x_hess_new)
            self.runtimes[self.k] = MPI.Wtime() - start

        # Final quantities to be assigned to self
        self.x = x
        self.logw = logw
        if self.rank==0:
        	self.diagnose.save_final_info(self.mean_estimate, self.mean_estimate_rc, 
                                        self.var_estimate, self.var_estimate_rc, 
                                        self.Neff, self.resampling_points, self.N,
                                        self.runtimes)
             
    def update_weights(self, x, x_new, v, v_half, logw, p_logpdf_x,
                       p_logpdf_x_new, grad, grad_new, hess, hess_new):
        
        if self.q.prop == 'rw':
            v_new = v_half
        elif self.q.prop == 'first_order':
            v_new = 0.5 * self.q.step_size*grad_new + v_half
        elif self.q.prop == 'second_order':
            # check degeneracy
            neg_hess = -hess_new
            if self.q.isPSD(neg_hess):
                v_new = 0.5*self.q.step_size*grad_new + v_half
            else:
                v_new = 0.5 * self.q.ss_2nd_1st*grad_new + v_half

        # Initialise
        logw_new = np.vstack(np.zeros(self.loc_n))

        # Use the forwards proposal as the L-kernel
        if self.optL == 'forwards-proposal':
            for i in range(self.loc_n):
                logw_new[i] = (logw[i] +
                               p_logpdf_x_new[i] -
                               p_logpdf_x[i] +
                               self.q.logpdf(x[i], x_new[i], -v_new[i], grad_new[i], hess_new[i]) - # lkernel
                               self.q.logpdf(x_new[i], x[i], v[i], grad[i], hess[i])) 
                
        return logw_new
