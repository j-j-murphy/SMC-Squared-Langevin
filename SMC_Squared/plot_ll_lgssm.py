import numpy as np
import csv
import sys
sys.path.append('..')  # noqa
from numpy.random import randn, choice
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI

from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from scipy.stats import uniform as Uniform_PDF
from scipy.stats import gamma, norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.special import logsumexp

import autograd
import torch
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd.functional import hessian

from SMCsq_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from SMC_DIAGNOSTICS import smc_no_diagnostics, smc_diagnostics_final_output_only, smc_diagnostics

from RNG import RNG

torch.manual_seed(42)
def generateData(theta, noObservations, initialState):
    mu = theta[0]
    phi = theta[1]
    sigmav = theta[2]

    state = np.zeros(noObservations + 1)
    observation = np.zeros(noObservations)
    state[0] = initialState

    for t in range(1, noObservations):
        state[t] = torch.distributions.Normal(mu * state[t-1], phi).sample() # mu * state[t - 1] + phi * np.random.randn()
        observation[t] = torch.distributions.Normal(state[t], sigmav).sample()

    return(state, observation)

parameters = np.zeros(3)    # theta = (phi, sigmav, sigmae)
parameters[0] = 0.75
parameters[1] = 1.2
parameters[2] = 1.2
noObservations = 1000
N_x = 8192
initialState = 0 

state, observations = generateData(parameters, noObservations, initialState)

class Target_PF():
    
    def __init__(self, N_x, y, prop, prior_logpdf, diag_hessian=True):
        self.y = y
        self.prop = prop
        self.diag_hessian = diag_hessian
        self.prior_logpdf = prior_logpdf
        self.N_x = N_x
    
    """ Define target """
    def logpdf(self, thetas, rngs):
        thetas_ = torch.tensor(thetas, requires_grad=True)

        try:
            LL = self.run_particleFilter(thetas_, rngs)
            grads = np.zeros(len(thetas))
            grads2 = np.zeros((len(thetas), len(thetas)))

            if self.prop == 'first_order' or self.prop == 'second_order':
                first_derivative = grad(LL, thetas_, create_graph=True)[0]
                grads = first_derivative.detach().numpy()#+ grad_prior_mu_first.detach().numpy()
                grads[np.isnan(grads)] = -np.inf
            
                if self.prop == 'second_order':
                    second_derivative = [torch.autograd.grad(first_derivative[i], thetas_, create_graph=True)[0].detach().numpy() for i in range(len(thetas_))]
                    second_derivative = np.stack(second_derivative)

                    if self.diag_hessian:
                        for i in range(len(thetas_)):
                            for j in range(len(thetas_)):
                                if i != j:
                                    second_derivative[i][j] = 0

                    second_derivative = second_derivative #+ grad_prior_mu_second.detach().numpy()
                    grads2 = second_derivative
                    grads2[np.isnan(grads2)] = -np.inf


            LL_=LL.detach().numpy()+self.prior_logpdf(thetas)

            # if np.any(np.isnan(first_derivative)) or np.any(np.isnan(second_derivative)) or torch.isnan(LL):
            #     print("here")
            #     test = np.array([-np.inf, -np.inf])
            #     test1 = np.array([[-np.inf, -np.inf],
            #                       [-np.inf, -np.inf]])
            #     LL_ = -np.inf
           

        except Exception as e:
            print(e)
            grads = np.full(len(thetas), -np.inf)
            grads2 = np.full((len(thetas), len(thetas)), -np.inf)
            LL_ = -np.inf

        return(LL_, grads, grads2)

    def run_particleFilter(self, thetas, rngs):
        mu = thetas[0]
        phi = thetas[1]
        sigmav = thetas[2]
        #sigmav = 1.2

        T = len(self.y)
        P = self.N_x

        xp = torch.zeros((T, P))
        lw = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        xp[0] = torch.full((1,P),  0)[0]+rngs.torchRandn(P)
        lw[:] = -torch.log(torch.ones(1,1)*P)
        noise = rngs.torchNormalrsample(torch.tensor([T, P]))
        resampletot = 0

        for t in range(1,T):

            xp[t] =  mu * xp[t-1].clone() + phi * noise[t]
            lognewWeights = lw.clone() + torch.distributions.Normal(xp[t], sigmav).log_prob(torch.tensor([self.y[t]]))
            lw = lognewWeights.clone()
            loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)
            wnorm = torch.exp(lw-loglikelihood[t]) #normalised weights (on a linear scale)
            neff = 1./torch.sum(wnorm*wnorm)

            if(neff<P/2):
                resampletot = resampletot + 1
                idx = rngs.torchMultinomial(P, wnorm)
                xp[:] = xp[:, idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)

        return(loglikelihood[T-1])


plt.rcParams['axes.grid'] = True
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['hatch.linewidth'] = 1.0
plt.rcParams["legend.frameon"] = 'True'
plt.rcParams["legend.fancybox"] = 'True'

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5,5),  constrained_layout=True)
fig.supylabel('loglikelihood', size=16)
axs[0].set_ylabel("$E[\mu]$")
axs[1].set_ylabel("$E[\phi]$")
axs[2].set_ylabel("E$[\sigma_v]$")

mus = np.linspace(0.65, 0.85, 31)
gammas = np.linspace(1.1, 1.3, 31)
sigmavs = np.linspace(1.1, 1.3, 31)
thetas = {"mu": mus, "gamma": gammas, "sigmav": sigmavs}


LL_all = []
  
p = Target_PF(N_x, observations, 0, 0)
rng = RNG(0)
for k in range(len(thetas)):
    ll = []
    thetas_list = list(thetas.keys())

    if thetas_list[k] == "mu":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([x, 1.2, 1.2]), rng))

    if thetas_list[k] == "gamma":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([0.75, x, 1.2]), rng)) 

    if thetas_list[k] == "sigmav":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([0.75, 1.2, x]), rng))

    axs[k].plot(thetas[thetas_list[k]], ll, color="b")
    axs[k].set_xlim([thetas[thetas_list[k]][0], thetas[thetas_list[k]][-1]])
    LL_all.append(ll)


fpath = f"outputs/z_ll/lgssm_Nx_{N_x}_T_{noObservations}"

plt.savefig(f"{fpath}.png")

with open(f'{fpath}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(LL_all)

# with open('lgssm_ll.csv', 'r') as read_obj: 
#     csv_reader = csv.reader(read_obj) 
#     LL_all = list(csv_reader) 