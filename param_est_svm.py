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

from models.lgssm_gradients_optimal import generateData, lgssm_PF

from smc_components.proposals import Q
from smc_components.SMCsq_BASE import SMC
from smc_components.SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from smc_components.SMC_DIAGNOSTICS import smc_diagnostics_final_output_only

class Q0(Q0_Base):
    def __init__(self):
        self.uni_1_pdf = Uniform_PDF(loc=-1, scale=2) #1
        self.uni_2_pdf = Uniform_PDF(loc=0, scale=1) #2
        self.uni_3_pdf = Uniform_PDF(loc=0, scale=1) #2

    def logpdf(self, x):
        return self.uni_1_pdf.logpdf(x[0]) + self.uni_2_pdf.logpdf(x[1]) + self.uni_3_pdf.logpdf(x[2])

    def rvs(self, rngs):
        return np.array([rngs.randomUniform(-1, 1)[0], 
                         rngs.randomUniform(0, 1)[0], 
                         rngs.randomUniform(0, 1)[0]])

# Data Generation params    
parameters = np.zeros(3) 
parameters[0] = 0.9
parameters[1] = 0.15
parameters[2] = 0.15
noObservations = 500
initialState = 0
seeds = np.arange(1, 50) 

# SMC2 params
N = 32#64
K = 15#20
D = 3
P=500
model = f"svm_N_{N}_K_{K}_D_{D}_P_{P}"
l_kernels = ['forwards-proposal']

# Prior
q0 = Q0()

sec_order_ss = np.linspace(0.7, 1.7, 21)
first_order_ss = np.linspace(0.02, 0.03, 21)#np.linspace(0.01, 0.02, 21)
rw_ss = np.linspace(0.1, 0.6, 21)
# first_order_ss = [0.085]
# sec_order_ss = [1.0]
# rw_ss = [0.175]
prop_step_sizes=  {'first_order':first_order_ss}#'rw': rw_ss,}#'second_order': sec_order_ss}# 'rw': rw_ss,#'rw': rw_ss, 'first_order':first_order_ss}#'second_order': sec_order_ss}# 'rw': rw_ss, 'first_order':first_order_ss}#, 'second_order': sec_order_ss}#

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
            state, observations = generateData(parameters, noObservations, initialState)
            p = lgssm_PF(q0.logpdf, observations, proposal, P)
            for step_size in step_sizes:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Running {proposal} with {l_kernel} kernel and step size {step_size} and seed {seed}")
                q = Q(step_size, proposal)
                diagnose = smc_diagnostics_final_output_only(model=model, proposal=proposal, l_kernel=l_kernel, step_size=step_size, seed=seed)
                diagnose.make_run_folder()
                smc = SMC(N, D, p, q0, K, proposal=q, optL=l_kernel, seed=42, rc_scheme='ESS_Recycling', verbose=True, diagnose=diagnose)
                smc.generate_samples()
