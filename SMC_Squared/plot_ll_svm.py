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

    x_t = torch.zeros(noObservations+1)
    y_t = torch.zeros(noObservations)
    x_t[0] = initialState

    for c in range(1, noObservations):
        x_t[c] = mu + phi * (x_t[c-1] - mu) + sigmav + torch.randn(1)
        y_t[c] = torch.distributions.Normal(0, torch.exp(x_t[c]/2)).sample()

    return(x_t, y_t)

parameters = np.zeros(3)    # theta = (phi, sigmav, sigmae)
parameters[0] = 0.5
parameters[1] = 0.9
parameters[2] = 0.2
noObservations = 128
initialState = 0 

state, observations = generateData(parameters, noObservations, initialState)

class Target_PF():
    
    def __init__(self, y, prop, prior_logpdf, diag_hessian=True):
        self.y = y
        self.prop = prop
        self.diag_hessian = diag_hessian
        self.prior_logpdf = prior_logpdf
    
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

        T = len(self.y)
        P = 5012

        xp = torch.zeros(P)
        lw = torch.zeros(P)
        lnorm = torch.zeros(P)
        proposalmean = torch.zeros(P)
        xp_new = torch.zeros(P)
        XtGivenXtMinus1Theta = torch.zeros(P)
        YtGivenXt = torch.zeros(P)
        proposalpdf = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        xp[:] = torch.full((1,P),  0.001)[0]+torch.randn(P)
        lw[:] = -torch.log(torch.ones(1,1)*P)
        noise = rngs.torchNormalrsample(torch.tensor([T, P]))
        resampletot = 0

        for t in range(1,T):
            xp_new = mu + phi * (xp.clone() - mu) + sigmav * noise[t]
            lognewWeights = lw.clone() + torch.distributions.Normal(0, torch.exp(xp_new.clone()/2)).log_prob(self.y[t-1])
            lw = lognewWeights.clone()
            loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)
            wnorm = torch.exp(lw-loglikelihood[t]) #normalised weights (on a linear scale)
            neff = 1./torch.sum(wnorm*wnorm)
            
            if(neff<P/2):
                resampletot = resampletot + 1
                idx = rngs.torchMultinomial(P, wnorm)
                xp= xp_new[idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)

        return(loglikelihood[T-2])


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

mus = np.linspace(0.45, 0.55, 21)
phis = np.linspace(0.85, 0.95, 21)
sigmavs = np.linspace(0.15, 0.25, 21)
thetas = {"mu": mus, "phi": phis, "sigmav": sigmavs}


LL_all = []
  


p = Target_PF(observations, 0, 0)
rng = RNG(0)
for k in range(len(thetas)):
    ll = []
    thetas_list = list(thetas.keys())

    if thetas_list[k] == "mu":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([x, 0.9, 0.2]), rng))

    if thetas_list[k] == "phi":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([0.5, x, 0.2]), rng)) 

    if thetas_list[k] == "sigmav":
        for x in thetas[thetas_list[k]]:
            ll.append(p.run_particleFilter(np.array([0.5, 0.9, x]), rng))

    axs[k].plot(thetas[thetas_list[k]], ll, color="b")
    axs[k].set_xlim([thetas[thetas_list[k]][0], thetas[thetas_list[k]][-1]])
    LL_all.append(ll)

plt.savefig("outputs/svm_ll.png")

with open('svm_ll.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(LL_all)

with open('svm_ll.csv', 'r') as read_obj: 
    csv_reader = csv.reader(read_obj) 
    LL_all = list(csv_reader) 