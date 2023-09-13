import numpy as np
import sys
sys.path.append('..')  # noqa
from SMCsq_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from numpy.random import randn, choice
from scipy.stats import gamma, norm, uniform
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import poisson
from scipy.special import logsumexp
from mpi4py import MPI
import warnings
from pathlib import Path
from SMC_DIAGNOSTICS import smc_no_diagnostics, smc_diagnostics_final_output_only, smc_diagnostics

warnings.simplefilter("error", "RuntimeWarning")

def simulate_data_SIR(N, t_length, beta, gamma_1, gamma_2, delta, E_initial):
   
    S_ = np.zeros(t_length)
    E_ = np.zeros(t_length)
    I_1_ = np.zeros(t_length)
    I_2_ = np.zeros(t_length)
    R_ = np.zeros(t_length)

    infected_ = np.zeros(t_length)
    infected_1 = np.zeros(t_length)
    deaths = np.zeros(t_length)
   
    S = N-E_initial #N-I_initial 0.9987
    E = E_initial
    I_1=1
    I_2=1
    R=0
    #D=0
    I_1_[0] = I_1
    I_2_[0] = I_2
    S_[0] = S - E - I_1 - I_2
    R_[0] = 0
    infected_[0] = 0
   
    zz = 1
    for t in range(1, t_length):
       
        p_SE  = 1 - np.exp(-beta * (I_1+I_2) / N) # S to I
        p_EI  = 1 - np.exp(-delta) # I to R
        p_I_1 = 1 - np.exp(-gamma_1) # I to R
        p_I_2 = 1 - np.exp(-gamma_2) # I to R

        n_SE = binom(S, p_SE).rvs()
        n_EI = binom(E, p_EI).rvs()
        n_I_1 = binom(I_1, p_I_1).rvs()
        n_I_2 = binom(I_2, p_I_2).rvs()

           
        S = S - n_SE
        E +=  n_SE - n_EI
        I_1 +=  n_EI - n_I_1
        I_2 +=  n_I_1 - n_I_2
        R +=  n_I_2
        #D +=  n_ID
        
        S_[t] = S
        E_[t] = E
        I_1_[t] = I_1
        I_2_[t] = I_2
        R_[t] = R
        #D_[t] = D

        infected_[zz] = poisson(I_1).rvs()
        
        infected_1[zz] = poisson(I_2).rvs()

        #deaths[zz] = poisson(D).rvs()
        zz +=1
       
    return(infected_, infected_1, S_, E_, I_1_, I_2_, R_)



"""
Testing for SMC_BASE
P.L.Green
"""


class Target_PF():
   
    def __init__(self, obs):
        self.obs = obs
   
    """ Define target """
    def logpdf(self, x, rngs):
        return(self.run_particleFilter(x, rngs))
   
   
    def run_particleFilter(self, thetas, rngs):
        
        obs_1 = self.obs[0]
        obs_2 = self.obs[1]
        
        beta = thetas[0]
       
        gamma_1 = thetas[1]
        
        gamma_2 = thetas[2]
       
        delta = thetas[3]
        
        if beta < 0:
            return(-np.inf)
            
        if gamma_1 < 0:
            return(-np.inf)
            
        if gamma_2 < 0:
            return(-np.inf)
            
        if delta < 0:
            return(-np.inf)
       

        P = 500
        T = len(obs_1)
        lw = np.zeros((T, P))
        lnorm = np.zeros(P)
        N = 10000
        S = np.zeros((T, P)).astype(int)
        E = np.zeros((T, P)).astype(int)
        I_1 = np.zeros((T, P)).astype(int)
        I_2 = np.zeros((T, P)).astype(int)
        R = np.zeros((T, P)).astype(int)
        #D = np.zeros((T, P)).astype(int)

        E[0] = np.full((1, P),3)[0]
        I_1[0] = np.ones(P)
        I_2[0] = np.ones(P)

        S[0] = N - E[0] - I_1[0] - I_2[0]
        #I[0]=I[0]
        R[0] = np.zeros(P)

        loglikelihood = np.zeros(T)

        lw[0] = -np.log(P)

        resampletot=0
        neff_list = []

        II = np.zeros(T)
        current_time = 0
        zz=1
        for t in range(1, T-1):
            if t > current_time:
                current_time += 1
            
                p_SI = 1 - np.exp(-beta * (I_1[current_time-1]+I_2[current_time-1]) / N) # S to I
                p_EI = 1 - np.exp(-delta) # I to R
                p_I_1 = 1 - np.exp(-gamma_1) # I to R
                p_I_2 = 1 - np.exp(-gamma_2) # I to R
                
                n_SI = rngs.randomBinomial(S[current_time-1], p_SI.astype(float))
                n_SE = rngs.randomBinomial(E[current_time-1], p_EI)
                n_I_1 = rngs.randomBinomial(I_1[current_time-1], p_I_1)
                n_I_2 = rngs.randomBinomial(I_2[current_time-1], p_I_2)
               
                S[current_time] = S[current_time-1] - n_SI
                E[current_time] = E[current_time-1] +  n_SI - n_SE
                I_1[current_time] = I_1[current_time-1] +  n_SE - n_I_1
                I_2[current_time] = I_2[current_time-1] +  n_I_1 - n_I_2
                R[current_time] = R[current_time-1] + n_I_2
                
            lw[t] = lw[t-1] + poisson.logpmf(I_1[current_time], obs_1[current_time]+0.00001) + poisson.logpmf(I_2[current_time], obs_2[current_time]+0.00001)

            loglikelihood[t] = logsumexp(lw[t])
            
            wnorm = np.exp(lw[t]-loglikelihood[t]) #normalised weights (on a linear scale)
            
            if np.isnan(wnorm).any():
                return(-np.inf) #normalised weights (on a linear scale)

            neff = 1./np.sum(wnorm*wnorm)
            neff_list.append(neff)
            zz +=1
            if(neff<P/2):
                resampletot = resampletot + 1
                idx = rngs.randomChoice(P, P, wnorm)
                S[:] = S[:, idx]
                E[:] = E[:, idx]
                I_1[:] = I_1[:, idx]
                I_2[:] = I_2[:, idx]
                R[:] = R[:, idx]
                lw[t] = loglikelihood[t]-np.log(P)
            
               
        return(loglikelihood[t-1])
       

class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.gauss_pdf = Normal_PDF(mean=np.zeros(1), cov=np.eye(1))
        self.gamma_pdf_0 = Gamma_PDF(a=2, scale=1)
        self.uniform_logpdf = uniform(0, 1)
        self.uniform_logpdf_1 = uniform(0, 0.5)

    def logpdf(self, x):
        return self.uniform_logpdf_1.logpdf(x[0]) + self.uniform_logpdf_1.logpdf(x[1]) + self.uniform_logpdf_1.logpdf(x[2]) + self.uniform_logpdf_1.logpdf(x[3])

    def rvs(self, size, rngs):
        return np.array([rngs.randomUniform(0, 0.5, size=1)[0], rngs.randomUniform(0, 0.5, size=1)[0], rngs.randomUniform(0, 0.5, size=1)[0], rngs.randomUniform(0, 0.5, size=1)[0]])


class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond, rngs):
        #print("&&&&&&&&&&&")
        #print(rngs.randomNormal(mu=0, sigma=1, size=len(x_cond)))
        return x_cond + 0.1 * rngs.randomNormal(mu=0, sigma=1, size=len(x_cond))


# No. samples and iterations
K = 50
D=4
optLs = ["gauss", "forwards-proposal"]#, "forwards-proposal"]
n_samples = [256, 512, 1024, 2048]#, 4096]#, 512, 1024]
times = []
seeds=range(0,10)



fig, ax = plt.subplots(ncols=2)
obs_list = []
II =[]
for ii in range(len(seeds)):
    np.random.seed(seeds[ii])
    N = 10000
    t_length = 50
    beta = 0.9
    gamma_1 = 0.08
    gamma_2 = 0.1
    delta = 0.3

    Exposed_initial = 3

    observations, observations_1, S, E, I_1, I_2, R= simulate_data_SIR(N, t_length, beta, gamma_1, gamma_2, delta, Exposed_initial)
    obs = [observations, observations_1]

    obs_list.append(obs)
    ax[0].plot(observations)
    ax[1].plot(observations_1)

plt.savefig("observations_SEIIR.png")

for optL in optLs:
    for samples in n_samples:
        for seed in seeds:

            N = samples
            p = Target_PF(obs_list[seed])
            q0 = Q0()
            q = Q()
            diagnose = smc_diagnostics_final_output_only(num_particles=N, num_cores=MPI.COMM_WORLD.Get_size(), seed=seed, l_kernel=optL, model="SEIIR")
            smc = SMC(N, D, p, q0, K, proposal=q, optL=optL, seed=seed, rc_scheme='ESS_Recycling', diagnose=diagnose) # forwards-proposal, gauss
            
            start = MPI.Wtime()
            smc.generate_samples()
            end = MPI.Wtime()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Runtime:", end-start)
                print("Cores:", MPI.COMM_WORLD.Get_size())
                print("Particles:", N)
                print("Seed:", seed)
                print(diagnose.fpath)
                f = open(Path(diagnose.fpath, "runtime.txt"), "w")
                f.write(str(end-start))
                f.close()


# n_samples = [512]

# times = []

# for samples in n_samples:
#     N = samples
#     print("samples = {}".format(samples))
#     p = Target_PF(obs)
#     q0 = Q0()
#     q = Q()
#     seed=30
#     smc = SMC(N, D, p, q0, K, proposal=q, optL='forwards-proposal', seed=seed, rc_scheme='ESS_Recycling')
    
#     start = MPI.Wtime()
#     smc.generate_samples()
#     end = MPI.Wtime()
#     if MPI.COMM_WORLD.Get_rank() == 0:
#     	times.append(end-start)
  
# print(times)

# fig, ax = plt.subplots(ncols=4)
                       
# ax[0].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 0], yerr=np.sqrt(smc.var_estimate_rc[:, 0, 0]), color='b', alpha=0.5)

# ax[1].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 1], yerr=np.sqrt(smc.var_estimate_rc[:, 1, 1]), color='b', alpha=0.5)

# ax[2].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 2], yerr=np.sqrt(smc.var_estimate_rc[:, 2, 2]), color='b', alpha=0.5)

# ax[3].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 3], yerr=np.sqrt(smc.var_estimate_rc[:, 3, 3]), color='b', alpha=0.5)
                                             
# ax[0].plot(np.repeat(0.3, K), 'lime', linewidth=3.0,
#                linestyle='--')
               
# ax[1].plot(np.repeat(0.08, K), 'lime', linewidth=3.0,
#                linestyle='--')
               
# ax[2].plot(np.repeat(0.1, K), 'lime', linewidth=3.0,
#                linestyle='--')

# ax[3].plot(np.repeat(0.3, K), 'lime', linewidth=3.0,
#                linestyle='--')


# plt.savefig("SEIIR_results_forwards.png")

#beta = 0.3
#gamma_1 = 0.08
#gamma_2 = 0.1
#delta = 0.3



