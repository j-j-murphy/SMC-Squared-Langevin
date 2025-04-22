import numpy as np
import sys
sys.path.append('..')  # noqa
from numpy.random import randn, choice
import matplotlib.pyplot as plt
from mpi4py import MPI
import torch

from scipy.stats import multivariate_normal as Normal_PDF
from scipy.stats import gamma as Gamma_PDF
from scipy.stats import uniform as Uniform_PDF

from models.sir import generateData, sir_PF

from smc_components.proposals import Q
from smc_components.SMCsq_BASE import SMC
from smc_components.SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from smc_components.SMC_DIAGNOSTICS import smc_diagnostics_final_output_only

class Q0(Q0_Base):
    def __init__(self):
        self.uni_1_pdf = Uniform_PDF(loc=0, scale=1) #1
        self.uni_2_pdf = Uniform_PDF(loc=0, scale=1) #2

    def logpdf(self, x):
        return self.uni_1_pdf.logpdf(x[0]) + self.uni_2_pdf.logpdf(x[1])

    def rvs(self, rngs):
        return np.array([rngs.randomUniform(0, 1)[0], 
                         rngs.randomUniform(0, 1)[0]])

# Data Generation params    
t_length = 35
beta = 0.6
gamma = 0.3
I_initial = 1
Npop = 763
seeds = np.arange(1, 50) 

# SMC2 params
N = 32#64
K = 15#20
D = 2
P=500
model = f"sir_N_{N}_K_{K}_D_{D}_P_{P}"
l_kernels = ['forwards-proposal']

# Prior
q0 = Q0()

sec_order_ss = np.linspace(1.7, 2.7, 21)
first_order_ss = np.linspace(0.007, 0.009, 21)
rw_ss = np.linspace(0.1, 1.1, 21)
# first_order_ss = [0.006]
# sec_order_ss = [0.8]
# rw_ss = [0.01]
prop_step_sizes=  {'first_order':first_order_ss}#'second_order': sec_order_ss}#, #'first_order':first_order_ss}##'rw': rw_ss, 'first_order':first_order_ss}#'second_order': sec_order_ss}# 'rw': rw_ss, 'first_order':first_order_ss}#, 

print(f"Running {model}")
if MPI.COMM_WORLD.Get_rank() == 0:
    print("Plotting info")
    print(f"models: {model}")
    print(f"prop_step_sizes: {prop_step_sizes}")
    print(f"l_kernels: {l_kernels}")
    print(f"seeds: {seeds}")
    print(f"Number of particles: {P}")
    print(f"Number of samples: {N}")

for proposal in prop_step_sizes.keys():
    step_sizes = prop_step_sizes[proposal]
    for l_kernel in l_kernels:
        for seed in seeds:
            torch.manual_seed(seed)
            observations, S, I = generateData(Npop, t_length, beta, gamma, I_initial)
            p = sir_PF(q0.logpdf, observations, Npop, proposal, P)
            for step_size in step_sizes:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Running {proposal} with {l_kernel} kernel and step size {step_size} and seed {seed}")
                q = Q(step_size, proposal)
                diagnose = smc_diagnostics_final_output_only(model=model, proposal=proposal, l_kernel=l_kernel, step_size=step_size, seed=seed)
                diagnose.make_run_folder()
                smc = SMC(N, D, p, q0, K, proposal=q, optL=l_kernel, seed=42, rc_scheme='ESS_Recycling', verbose=True, diagnose=diagnose)
                smc.generate_samples()
