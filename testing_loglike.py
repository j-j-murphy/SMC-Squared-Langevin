import numpy as np
import sys
sys.path.append('..')  # noqa
from numpy.random import randn, choice
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from scipy.stats import uniform as Uniform_PDF
from scipy.stats import gamma, norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.special import logsumexp
from smc_components.RNG import RNG

import autograd
import torch
from torch.autograd import Variable
from torch.autograd import grad
from torch.autograd.functional import hessian

from smc_components.SMCsq_BASE import SMC
from smc_components.SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from smc_components.SMC_DIAGNOSTICS import smc_no_diagnostics, smc_diagnostics_final_output_only, smc_diagnostics

from models.lgssm_gradients_optimal import Target_PF, generateData

parameters = np.zeros(3)    # theta = (phi, sigmav, sigmae)
parameters[0] = 0.75
parameters[1] = 1.
parameters[2] = 1.
noObservations = 500
initialState = 0
phis = np.linspace(0.0, 1.5, 50)
seeds = np.arange(1, 4)

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].set_xlabel("phis")
ax[0].set_ylabel("LL")
ax[1].set_xlabel("phis")
ax[1].set_ylabel("Gradient")
ax[2].set_xlabel("phis")
ax[2].set_ylabel("2nd Derivative")

for seed in seeds:
    torch.manual_seed(seed)
    state, observations = generateData(parameters, noObservations, initialState)

    p = Target_PF(observations, "second_order", 1000)

    grads = []
    LLL= []
    second_order_derivatives = []
    rngs = RNG()
    # 1 d gaussian
    # gauss = torch.distributions.Normal(0, 1)
    for mu in phis:
        # ### 1D Gaussian Test ###
        # thetas = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
        # LL = gauss.log_prob(thetas)
        # LLL.append(LL.item())
        # first_derivative = torch.autograd.grad(LL, thetas, create_graph=True)[0]
        # grads.append(first_derivative.item())
        # second_derivative = torch.autograd.grad(first_derivative, thetas)[0]
        # second_order_derivatives.append(second_derivative.item())
        print(f"Processing mu = {mu}")
        
        # Create tensor with requires_grad=True in one step
        thetas = torch.tensor([mu], dtype=torch.float64, requires_grad=True)
        
        # Compute log-likelihood
        LL = p.run_particleFilter(thetas, rngs)  # Log-likelihood
        LLL.append(LL.item())  # Append LL directly as a Python float

        # Compute first-order derivative
        first_derivative = grad(LL, thetas, create_graph=True)[0]
        grads.append(first_derivative.item())  # Append first derivative as a float

        # Compute second-order derivative
        second_derivative = grad(first_derivative, thetas)[0]
        second_order_derivatives.append(second_derivative.item())  # Append second derivative as a float

    # Plot the log-likelihood
    ax[0].plot(phis, LLL, label="Log-Likelihood")
    
    # Plot the gradient (first-order derivative)
    ax[1].plot(phis, grads, label="Gradient (1st Derivative)")

    # Plot the second-order derivative
    ax[2].plot(phis, second_order_derivatives, label="2nd Derivative")


# plt.legend()
plt.tight_layout()
plt.savefig("loglike.png")
