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


def simulate_data_SIR(N, t_length, beta, gamma, I_initial):
   
    S_ = np.zeros(t_length)
    I_ = np.zeros(t_length)
    R_ = np.zeros(t_length)
    infected_ = np.zeros(t_length)
   
    S = N #N-I_initial 0.9987
    I=I_initial
    R=0
    I_[0] = I
    S_[0] = S
    R_[0] = 0
    infected_[0] = 3
   
    for t in range(1, t_length):
       
        p_SI = 1 - np.exp(-beta * I / N) # S to I
        p_IR = 1 - np.exp(-gamma) # I to R

        n_SI = binom(S, p_SI).rvs()
        n_IR = binom(I, p_IR).rvs()

           
        S = S - n_SI
        I +=  n_SI - n_IR
        R +=  n_IR
       
        I_[t] = I
        S_[t] = S
        R_[t] = R

        infected_[t] = poisson(I).rvs() # nbinom(I, 0.8).rvs()
       
    return(infected_, S_, I_, R_)

#plt.plot(S)
#plt.plot(I)
#plt.plot(R)
#plt.plot(observations)
#plt.savefig("observations_SIR_pmcmc.png")


class Target_PF():
   
    def __init__(self, obs):
        self.obs = obs
   
    """ Define target """
    def logpdf(self, x, rngs_pf):
        return(self.run_particleFilter(x, rngs_pf))
   
    def run_particleFilter(self, thetas, rngs):
       
        beta = thetas[0]
       
        gamma = thetas[1]
        
        if beta < 0 or gamma < 0:
            return(-np.inf)
        

        P = 500
        T = len(self.obs)
        lw = np.zeros((T, P))
        lnorm = np.zeros(P)
        N = 10000 #10000#300 # 763
        S = np.zeros((T, P)).astype(int)
        I = np.zeros((T, P)).astype(int)
        R = np.zeros((T, P)).astype(int)

        I[0] = np.full((1, P),3)[0]

        S[0] = N - I[0]
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
                p_IR = 1 - np.exp(-gamma) # I to R  
                
                n_SI = rngs.randomBinomial(S[current_time-1], p_SI.astype(float))
                n_IR = rngs.randomBinomial(I[current_time-1], p_IR)

                S[current_time] = S[current_time-1] - n_SI
                I[current_time] = I[current_time-1] +  n_SI - n_IR
                R[current_time] = R[current_time-1] + n_IR

            lw[t] = lw[t-1] + poisson.logpmf(I[current_time], self.obs[current_time]+0.000001)     
            loglikelihood[t] = logsumexp(lw[t])    
            wnorm = np.exp(lw[t]-loglikelihood[t])
            
            if np.isnan(wnorm).any():
                return(-np.inf) #normalised weights (on a linear scale)
        
            neff = 1./np.sum(wnorm*wnorm)
            neff_list.append(neff)
            zz +=1
            if(neff<P/2):
                resampletot = resampletot + 1
                idx = rngs.randomChoice(P, P, wnorm)
                S[:] = S[:, idx]
                I[:] = I[:, idx]
                R[:] = R[:, idx]
                lw[t] = loglikelihood[t]-np.log(P)              
            
        return(loglikelihood[t-1])

class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        return -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond, rngs):
        return x_cond + 0.05 * rngs.randomNormal(mu=0, sigma=1, size=len(x_cond))
        
class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.gauss_pdf      = Normal_PDF(mean=np.zeros(1), cov=np.eye(1))
        self.gamma_pdf      = Gamma_PDF(a=2, scale=2/1)
        self.exp_logpmf     = expon(5)
        self.uniform_logpdf = uniform(0, 1)

    def logpdf(self, x):
        return self.uniform_logpdf.logpdf(x[0]) + self.uniform_logpdf.logpdf(x[1])

    def rvs(self, size, rngs):
        return np.array([rngs.randomUniform(0, 1, size=1)[0], rngs.randomUniform(0, 1, size=1)[0]])


seed = 1
problem_size = np.array([256, 512, 1024, 2048])*10
K= 2
seeds=range(0,10)

obs_list = []
II =[]
for ii in range(len(seeds)):
    np.random.seed(seeds[ii])
    N  = 10000
    t_length = 30
    beta = 0.85
    gamma = 0.2

    Infected_initial = 3

    observations, S, I, R = simulate_data_SIR(N, t_length, beta, gamma, Infected_initial)
    obs_list.append(observations)
    plt.plot(observations)
plt.savefig("observations_pmcmc_SIR.png")

fpath = Path("outputs/SIR_pmcmc")

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

        samples[0] = q0.rvs(2, rng)
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

fig, ax = plt.subplots(ncols=2)

ax[0].plot(samples[:, 0][500:])
ax[1].plot(samples[:, 1][500:])

ax[0].plot(np.repeat(0.85, M)[500:], 'lime', linewidth=3.0,
               linestyle='--')

ax[1].plot(np.repeat(0.2, M)[500:], 'lime', linewidth=3.0,
               linestyle='--')
plt.savefig("SIR_results_pmcmc.png")

def mse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()

print(mse(np.repeat(0.85, M)[500:], samples[:, 0][500:]))
print(mse(np.repeat(0.2, M)[500:], samples[:, 1][500:]))
        
    
    
    
    





