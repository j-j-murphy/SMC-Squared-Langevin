import itertools
from pathlib import Path
import numpy as np

SMCS_OUTPUT_DIR = Path("outputs")
model = ["SIR", "SEIR", "SEIIR"]
l_kernel = ["gauss", "forwards-proposal"]
particles = [256, 512, 1024, 2048, 4096]
cores = [1, 2, 4, 8, 16, 32, 64, 128]
seeds = range(0,10)

# hyperparams = [model, l_kernel, cores, particles, seeds]
# combinations = list(itertools.product(*hyperparams))

# for combo in combinations:
#     model, l_kernel, core, particle, seed = combo
#     output_dir = Path.joinpath(SMCS_OUTPUT_DIR, f"{model}_{l_kernel}", f"particles_{particle}", f"cores_{core}", f"seed_{seed}")
#     output_dir.mkdir(parents=True, exist_ok=True)

pmcmc_model = ["SIR_pmcmc"]
problem_size = np.array([256, 512, 1024, 2048])*10

hyperparams_pmcmc = [pmcmc_model, problem_size, seeds]
combinations_pmcmc = list(itertools.product(*hyperparams_pmcmc))

for combo in combinations_pmcmc:
    model, size, seed = combo
    output_dir = Path.joinpath(SMCS_OUTPUT_DIR, f"{model}", f"size_{size}", f"seed_{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

models = ["SIR_gauss"]
particles = [1024]
cores = [32]
seeds = range(0,10)
iterations = 10

for model in models:
    for particle in particles:
        for core in cores:
            for seed in seeds:
                for iteration in range(iterations):
                    output_dir = Path.joinpath(SMCS_OUTPUT_DIR, f"{model}", f"particles_{particle}", f"cores_{core}", f"seed_{seed}", f"iteration_{iteration}")
                    output_dir.mkdir(parents=True, exist_ok=True)
