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

def simulate_data_SIR(N, t_length, beta, gamma, delta, E_initial):
   
    S_ = np.zeros(t_length)
    E_ = np.zeros(t_length)
    I_ = np.zeros(t_length)
    R_ = np.zeros(t_length)
    infected_ = np.zeros(t_length)
   
    S = N #N-I_initial 0.9987
    E = E_initial
    I=0
    R=0
    I_[0] = I
    S_[0] = S - E
    R_[0] = 0
    infected_[0] = 3
   
    for t in range(1, t_length):
       
        p_SE = 1 - np.exp(-beta * I / N) # S to I
        p_EI = 1 - np.exp(-delta) # I to R
        p_IR = 1 - np.exp(-gamma) # I to R

        n_SE = binom(S, p_SE).rvs()
        n_EI = binom(E, p_EI).rvs()
        n_IR = binom(I, p_IR).rvs()

           
        S = S - n_SE
        E +=  n_SE - n_EI
        I +=  n_EI - n_IR
        R +=  n_IR
        
        S_[t] = S
        E_[t] = E
        I_[t] = I
        R_[t] = R

        infected_[t] = poisson(I).rvs()

    return(infected_, S_, E_, I_, R_)


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
        #np.random.seed(seed=rngs_pf)
        beta = thetas[0]
       
        gamma = thetas[1]
       
        delta = thetas[2]
        
        if beta < 0:
            return(-np.inf)
            
        if gamma < 0:
            return(-np.inf)
            
        if delta < 0:
            return(-np.inf)
       

        P = 500
        T = len(self.obs)
        lw = np.zeros((T, P))
        lnorm = np.zeros(P)
        N = 10000
        S = np.zeros((T, P)).astype(int)
        E = np.zeros((T, P)).astype(int)
        I = np.zeros((T, P)).astype(int)
        R = np.zeros((T, P)).astype(int)

        E[0] = np.full((1, P),3)[0]

        S[0] = N - E[0]
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
                
                p_SI = 1 - np.exp(-beta * I[current_time-1] / N) # S to I
                p_EI = 1 - np.exp(-delta) # I to R
                p_IR = 1 - np.exp(-gamma) # I to R
                n_SI = rngs.randomBinomial(S[current_time-1], p_SI.astype(float))
                n_SE = rngs.randomBinomial(E[current_time-1], p_EI)
                n_IR = rngs.randomBinomial(I[current_time-1], p_IR)
               

                S[current_time] = S[current_time-1] - n_SI
                E[current_time] = E[current_time-1] +  n_SI - n_SE
                I[current_time] = I[current_time-1] +  n_SE - n_IR
                R[current_time] = R[current_time-1] + n_IR
                
            lw[t] = lw[t-1] + poisson.logpmf(I[current_time], self.obs[current_time]+0.00001)


            loglikelihood[t] = logsumexp(lw[t])
            
            wnorm = np.exp(lw[t]-loglikelihood[t]) #normalised weights (on a linear scale)

            neff = 1./np.sum(wnorm*wnorm)
            neff_list.append(neff)
            zz +=1
            if(neff<P/2):
                resampletot = resampletot + 1
                idx = rngs.randomChoice(P, P, wnorm)
                S[:] = S[:, idx]
                E[:] = E[:, idx]
                I[:] = I[:, idx]
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
        return self.uniform_logpdf_1.logpdf(x[0]) + self.uniform_logpdf_1.logpdf(x[1]) + self.uniform_logpdf_1.logpdf(x[2])

    def rvs(self, size, rngs):
        return np.array([rngs.randomUniform(0, 0.5, size=1)[0], rngs.randomUniform(0, 0.5, size=1)[0], rngs.randomUniform(0, 0.5, size=1)[0]])


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
D=3
optLs = ["gauss", "forwards-proposal"]#, "forwards-proposal"]
n_samples = [256, 512, 1024, 2048]#, 4096]#, 512, 1024]
times = []
seeds=range(0,10)


# np.random.seed(10)
# N = 10000
# t_length = 220
# beta = 0.3
# gamma = 0.08
# delta = 0.3

# Exposed_initial = 3

# observations, S, E, I, R = simulate_data_SIR(N, t_length, beta, gamma, delta, Exposed_initial)


# plt.plot(S)
# plt.plot(E)
# plt.plot(I)
# plt.plot(R)
# plt.plot(observations)
# plt.savefig("observations_SEIR.png")


obs_list = []
II =[]
for ii in range(len(seeds)):
    np.random.seed(seeds[ii])
    N = 10000
    t_length = 50
    beta = 0.9
    gamma = 0.08
    delta = 0.3

    Exposed_initial = 3

    observations, S, E, I, R = simulate_data_SIR(N, t_length, beta, gamma, delta, Exposed_initial)

    obs_list.append(observations)
    plt.plot(observations)
plt.savefig("observations_SEIR.png")

for optL in optLs:
    for samples in n_samples:
        for seed in seeds:

            N = samples
            p = Target_PF(obs_list[seed])
            q0 = Q0()
            q = Q()
            diagnose = smc_diagnostics_final_output_only(num_particles=N, num_cores=MPI.COMM_WORLD.Get_size(), seed=seed, l_kernel=optL, model="SEIR")
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

# fig, ax = plt.subplots(ncols=3)
                       
# ax[0].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 0], yerr=np.sqrt(smc.var_estimate_rc[:, 0, 0]), color='b', alpha=0.5)

# ax[1].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 1], yerr=np.sqrt(smc.var_estimate_rc[:, 1, 1]), color='b', alpha=0.5)

# ax[2].errorbar(x=np.arange(0, K), y=smc.mean_estimate_rc[:, 2], yerr=np.sqrt(smc.var_estimate_rc[:, 2, 2]), color='b', alpha=0.5)
                       
                                             
# ax[0].plot(np.repeat(0.3, K), 'lime', linewidth=3.0,
#                linestyle='--')
               
# ax[1].plot(np.repeat(0.08, K), 'lime', linewidth=3.0,
#                linestyle='--')
               
# ax[2].plot(np.repeat(0.3, K), 'lime', linewidth=3.0,
#                linestyle='--')

# #beta = 0.3
# #gamma = 0.08
# #delta = 0.3


# plt.savefig("SEIR_results_forwards.png")





