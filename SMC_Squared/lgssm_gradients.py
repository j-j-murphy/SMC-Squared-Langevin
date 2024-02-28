import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--proposal', type=str, default='rw')
parser.add_argument('-start', '--start_step_size', type=float, default=0.1)
parser.add_argument('-num', '--num_steps', type=int, default=10)
parser.add_argument('-stride', '--step_size_stride', type=float, default=0.1)
args = parser.parse_args()

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
noObservations = 500
initialState = 0 

state, observations = generateData(parameters, noObservations, initialState)

plt.plot(observations)


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
        sigmav = torch.tensor([1.2])

        T = len(self.y)
        P = 150

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

class Q0(Q0_Base):
    """ Define initial proposal """

    def __init__(self):
        self.gauss_pdf = Normal_PDF(mean=np.zeros(1), cov=np.eye(1))
        self.gamma_pdf = Gamma_PDF(a=1, scale=1)
        self.uni_1_pdf = Uniform_PDF(loc=-1, scale=2)
        self.uni_2_pdf = Uniform_PDF(loc=0, scale=10)

    def logpdf(self, x):
        return self.uni_1_pdf.logpdf(x[0]) + self.uni_2_pdf.logpdf(x[1]) +self.uni_2_pdf.logpdf(x[2])

    def rvs(self, size, rngs):
        return np.array([rngs.randomUniform(-1, 1)[0], rngs.randomUniform(0, 10)[0], rngs.randomUniform(0, 10)[0]])


class Q(Q_Base):
    """ Define general proposal """
    def __init__(self, step_size, prop):
        self.step_size = step_size
        self.prop = prop

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond, v, grads_1):        
        if self.prop == 'first_order':
            logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=self.step_size**2 * np.eye(len(x_cond))).logpdf(v)
        
        if self.prop == 'second_order':
            grads_1 = -grads_1

            if self.isPSD(grads_1):
                cov = np.linalg.pinv(grads_1)#.flatten()
                cov_1 = self.step_size**2 * cov
                logpdf = Normal_PDF(mean=x_cond, cov=cov_1).logpdf(v)
            else:
                logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=self.step_size**2 * np.eye(len(x_cond))).logpdf(v)
            
        if self.prop == 'rw':
            logpdf = Normal_PDF(mean=x_cond, cov=self.step_size**2 * np.eye(len(x_cond))).logpdf(x)
        
        return logpdf
    

    def rvs(self, x_cond, rngs, grads, grads_1):
        if self.prop == 'first_order':
            v = rngs.randomMVNormal(np.zeros(len(x_cond)), self.step_size**2 * np.eye(len(x_cond)))
            x_new = x_cond +0.5 * self.step_size**2 * grads + v

        elif self.prop == 'second_order':
            grads_1 = -grads_1

            if self.isPSD(grads_1):
                cov = np.linalg.pinv(grads_1)#.flatten()
                cov_1 = self.step_size**2 * cov
                v = rngs.randomMVNormal(np.zeros(len(x_cond)), cov_1)
                x_new = x_cond + 0.5 * self.step_size**2 * np.dot(grads, cov) + v

            else:
                v = rngs.randomMVNormal(np.zeros(len(x_cond)), self.step_size**2 * np.eye(len(x_cond)))
                x_new = x_cond + 0.5 * self.step_size**2 * grads + v

        elif self.prop == 'rw':
            v = rngs.randomMVNormal(np.zeros(len(x_cond)), self.step_size**2 * np.eye(len(x_cond)))
            x_new = x_cond + v

        return x_new, v
    
    def isPSD(self, x):
        try:
            return np.all(np.linalg.eigvals(x) > 0)
        except:
            return False

# No. samples and iterations
N = 4
K = 5
D = 3

q0 = Q0()

model = f"lgssm_{N}_3d"
proposals = [args.proposal]#, 'first_order', 'rw']
l_kernels = ['gauss', 'forwards-proposal']
# step_sizes = np.linspace(1.0, 1.6, 61)
# step_sizes = np.linspace(0.03, 0.05, 21)
#step_sizes = np.linspace(0.45, 0.55, 11)
step_sizes = args.start_step_size + np.arange(0, args.num_steps) * args.step_size_stride
#step_sizes = np.linspace(1.0, 1.2, 21)
#seeds = np.arange(0, 5)
seeds = np.arange(0, 3)

if MPI.COMM_WORLD.Get_rank() == 0:
    print("Plotting info")
    print(f"models: {model}")
    print(f"proposals: {proposals}")
    print(f"l_kernels: {l_kernels}")
    print(f"step_sizes: {step_sizes}")
    print(f"seeds: {seeds}")

for proposal in proposals:
    for l_kernel in l_kernels:
        for step_size in step_sizes:
            for seed in seeds:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Running {proposal} with {l_kernel} kernel and step size {step_size} and seed {seed}")

                p = Target_PF(observations, proposal, q0.logpdf)
                q = Q(step_size, proposal)
                diagnose = smc_diagnostics_final_output_only(model=model, proposal=proposal, l_kernel=l_kernel, step_size=step_size, seed=seed)
                diagnose.make_run_folder()
                smc = SMC(N, D, p, q0, K, proposal=q, optL=l_kernel, seed=seed, rc_scheme='ESS_Recycling', verbose=True, diagnose=diagnose)
                smc.generate_samples()