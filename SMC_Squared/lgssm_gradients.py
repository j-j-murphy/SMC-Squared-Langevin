import numpy as np
import sys
sys.path.append('..')  # noqa
from numpy.random import randn, choice
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from scipy.stats import gamma, norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.special import logsumexp

import autograd
import torch
from torch.autograd import Variable
from torch.autograd import grad

from SMCsq_BASE import SMC
from SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from SMC_DIAGNOSTICS import smc_no_diagnostics, smc_diagnostics_final_output_only, smc_diagnostics


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
noObservations = 500
initialState = 0 

state, observations = generateData(parameters, noObservations, initialState)

plt.plot(observations)


class Target_PF():
    
    def __init__(self, y):
        self.y = y
    
    """ Define target """
    def logpdf(self, thetas, rngs):
        mu_ = Variable(torch.tensor(thetas[0]),requires_grad=True)
        try:

            first_derivative = grad(self.run_particleFilter(mu_, rngs), mu_, create_graph=True)[0]
            # We now have dloss/dx
            second_derivative = grad(first_derivative, mu_)[0]

            #thetas_ = torch.tensor([mu_])
            LL = self.run_particleFilter(mu_, rngs)
            #LL.backward()
            #grad_beta  = mu_.grad
                
            grad_prior_mu_first = grad(torch.distributions.Normal(0, 1).log_prob(mu_), mu_, create_graph=True)[0]
            grad_prior_mu_second = grad(grad_prior_mu_first, mu_)[0]
            #print(grad(torch.distributions.Gamma(1, 1).log_prob(mu_), mu_, create_graph=True)[0].detach().numpy())
                
            first_derivative  = first_derivative.detach().numpy() + grad_prior_mu_first.detach().numpy()
            second_derivative = second_derivative.detach().numpy() + grad_prior_mu_second.detach().numpy()
            test = np.array([first_derivative])
            test1 = np.array([[second_derivative]])
            LL_=LL.detach().numpy()
            
        except:
            test = np.array([-np.inf])
            test1 = np.array([-np.inf])
            LL_ = -np.inf


        return(LL_, test, test1)

    def run_particleFilter(self, mu, rngs):
        #run particle filter
        torch.manual_seed(rngs)
        T = len(self.y)
        P = 150
        phi = torch.tensor([1.2])
        sigmav = torch.tensor([1.2])
        xp = torch.zeros((T, P))
        lw = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        xp[0] = torch.full((1,P),  0)[0]+torch.randn(P)
        lw[:] = -torch.log(torch.ones(1,1)*P)

        noise = torch.distributions.Normal(0, 1).rsample(torch.tensor([T, P]))
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
                #hate_my_life = torch.nn.functional.gumbel_softmax(torch.log(wnorm).repeat(P, 1), tau=1, hard=True)
                #idx = np.where(hate_my_life[:]==1)[1]
                idx = torch.multinomial(wnorm, P, replacement=True)
                xp[:] = xp[:, idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)

        return(loglikelihood[T-1])

class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.gauss_pdf = Normal_PDF(mean=np.zeros(1), cov=np.eye(1))
        self.gamma_pdf = Gamma_PDF(a=1, scale=1)

    def logpdf(self, x):
        return self.gauss_pdf.logpdf(x[0])

    def rvs(self, size, rngs):
        return np.array([rngs.randomNormal(mu=0, sigma=1, size=1)])


class Q(Q_Base):
    """ Define general proposal """

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond):
        
        return -0.5 * (x - x_cond).T @ (x - x_cond)

    def rvs(self, x_cond, rngs, grads, grads_1, props):
        if props == 'first':
            phi = 0.05
            new_0 = x_cond + 0.5 * phi**2 * grads + np.random.multivariate_normal(np.zeros(len(x_cond)), phi**2 * np.eye(len(x_cond)))

        if props == 'second':
            phi = 1
            grads_1 = -grads_1

            if self.isPSD(np.array([[grads_1]])):
                cov = np.linalg.pinv(np.array([[grads_1]])).flatten()
                cov_1 = phi**2 * np.array([cov])
                new_01 = x_cond + 0.5 * phi**2 * np.dot(grads, cov) + np.random.multivariate_normal(np.zeros(len(x_cond)), cov_1)
                new_0 = new_01.flatten()

            else:
                new_0 = x_cond + 0.5 * phi**2 * grads + np.random.multivariate_normal(np.zeros(len(x_cond)), phi**2 * np.eye(len(x_cond)))

        if props == 'third':
            new_0 = x_cond + 0.3 * np.random.randn(1)

        return new_0
    
    def isPSD(self, x):
        return np.all(np.linalg.eigvals(x) > 0)


# No. samples and iterations
N = 8
K = 2
D=1

p = Target_PF(observations)
q0 = Q0()
q = Q()
optL = 'forwards-proposal'

seed=1
diagnose = smc_diagnostics_final_output_only(num_particles=N, num_cores=MPI.COMM_WORLD.Get_size(), seed=seed, l_kernel=optL, model="lgssm")
diagnose.make_run_folder()
test = []
proprr = ['third', 'second', 'first']
for idx, x in enumerate(range(3)):
    print(proprr[idx])
    smc = SMC(N, D, p, q0, K, proposal=q, optL=optL, seed=seed, prop=proprr[idx], rc_scheme='ESS_Recycling', verbose=True, diagnose=diagnose)
    smc.generate_samples()
    test.append(smc)