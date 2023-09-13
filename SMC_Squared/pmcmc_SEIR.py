import numpy as np
import sys
sys.path.append('..')  # noqa
from SMCsq_BASE import SMC
from RNG import RNG
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from numpy.random import randn, choice
import matplotlib.pyplot as plt
from scipy.stats import poisson, expon, uniform, binom, nbinom, gamma, norm
from scipy.special import logsumexp
from mpi4py import MPI
import time
import warnings
from pathlib import Path


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

np.random.seed(10)
N = 10000
t_length = 50
beta = 0.9
gamma = 0.08
delta = 0.3

Exposed_initial = 3

observations, S, E, I, R = simulate_data_SIR(N, t_length, beta, gamma, delta, Exposed_initial)


plt.plot(S)
plt.plot(E)
plt.plot(I)
plt.plot(R)
plt.plot(observations)
plt.savefig("observations_SEIR.png")

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
        return np.array([rngs.randomUniform(0, 1, size=1)[0], rngs.randomUniform(0, 1, size=1)[0], rngs.randomUniform(0, 1, size=1)[0]])


class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond, rngs):
        return x_cond + 0.1 * rngs.randomNormal(mu=0, sigma=1, size=len(x_cond))





problem_size = np.array([256, 512, 1024])*10
K= 3
seeds=range(0,10)

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
plt.savefig("observations_pmcmc_SEIR.png")

fpath = Path("outputs/SEIR_pmcmc")

for M in problem_size:
    for seed in seeds:
        print("run")
        t0 = time.time() 
        p = Target_PF(obs_list[seed])
        q0 = Q0()
        q = Q()
        rng = RNG(seed)
        samples = np.zeros([M, K])
        samples_proposed = np.zeros([M, K])
        LL = np.zeros(M)
        LL_proposed = np.zeros(M)

        samples[0] = q0.rvs(1, rng)
        LL[0] = p.logpdf(samples[0], rng)


        for k in range(1, M):
        
            theta_proposed = q.rvs(x_cond=samples[k-1], rngs=rng)
            LL_pro = p.logpdf(theta_proposed, rng)

            if (LL_pro == -np.inf) or (LL_pro == np.inf) or (np.isnan(LL_pro)):

                samples[k] = samples[k-1]
                LL[k] = LL[k-1]

            else:
                samples_proposed[k] = theta_proposed
                LL_proposed[k] = LL_pro
                LL_diff = LL_proposed[k] - LL[k - 1]
                acceptProbability = np.min((np.log(1.0), LL_diff))

                uniformRandomVariable = rng.randomUniform(0, 1, size=1)[0]
                if np.log(uniformRandomVariable) < acceptProbability:
                # Accept the parameter
                    samples[k] = samples_proposed[k]
                    LL[k] = LL_proposed[k]
                else:
                    # Reject the parameter
                    samples[k] = samples[k - 1]
                    LL[k] = LL_proposed[k - 1]

        t1 = time.time()        

        total = t1-t0
        print(total)
        f = open(Path(fpath, f"size_{M}", f"seed_{seed}", "runtime.txt"), "w")
        f.write(str(total))
        f.close()
        #np.savetxt(Path(fpath, f"size_{M}", f"seed_{seed}", "runtime.csv"), total, delimiter=",")
        np.savetxt(Path(fpath, f"size_{M}", f"seed_{seed}", "samples.csv"), samples, delimiter=",")
        

