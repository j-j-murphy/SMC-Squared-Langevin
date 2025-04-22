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

from models.earthquake import earthquake_PF

from smc_components.proposals import Q
from smc_components.SMCsq_BASE import SMC
from smc_components.SMC_TEMPLATES import Target_Base, Q0_Base, Q_Base
from smc_components.SMC_DIAGNOSTICS import smc_diagnostics_final_output_only

class Q0(Q0_Base):
    def __init__(self):
        self.uni_1_pdf = Uniform_PDF(loc=-1, scale=2) #1
        self.uni_2_pdf = Uniform_PDF(loc=0, scale=2) #2
        self.uni_3_pdf = Uniform_PDF(loc=0, scale=20) #2

    def logpdf(self, x):
        return self.uni_1_pdf.logpdf(x[0]) + self.uni_2_pdf.logpdf(x[1]) + self.uni_3_pdf.logpdf(x[2])

    def rvs(self, rngs):
        return np.array([rngs.randomUniform(-1, 1)[0], 
                         rngs.randomUniform(0, 2)[0], 
                         rngs.randomUniform(0, 20)[0]])

# Data Generation params    
observations = torch.tensor([13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18,
    25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26,
    13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26,
    28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20,
    15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
    18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
    15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11])

# SMC2 params
N = 32#64
K = 15#20
D = 3
P=500
model = f"earthquake_N_{N}_K_{K}_D_{D}_P_{P}"
l_kernels = ['forwards-proposal']
seeds = np.arange(1, 50) 

# Prior
q0 = Q0()

sec_order_ss = np.linspace(0.3, 0.8, 21)#np.linspace(0.07, 0.17, 21)
first_order_ss = np.linspace(0.06, 0.08, 21)
rw_ss = np.linspace(0.025, 0.075, 21)
# first_order_ss = [0.085]
# sec_order_ss = [1.5]
# rw_ss = [0.175]
prop_step_sizes=  {'first_order':first_order_ss}#'second_order': sec_order_ss}#'rw': rw_ss}#'}#}#, 'rw': rw_ss}#, '}# 'rw': rw_ss, 'first_order':first_order_ss}#, 'second_order': sec_order_ss}#

print("Running lgssm_gradients_optimal.py")
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
            p = earthquake_PF(q0.logpdf, observations, proposal, P)
            for step_size in step_sizes:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Running {proposal} with {l_kernel} kernel and step size {step_size} and seed {seed}")
                q = Q(step_size, proposal)
                diagnose = smc_diagnostics_final_output_only(model=model, proposal=proposal, l_kernel=l_kernel, step_size=step_size, seed=seed)
                diagnose.make_run_folder()
                smc = SMC(N, D, p, q0, K, proposal=q, optL=l_kernel, seed=seed, rc_scheme='ESS_Recycling', verbose=True, diagnose=diagnose)
                smc.generate_samples()
