import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Base directory and model components setup
base_dir = 'outputs'
models = ["lgssm_32_explore"]
l_kernels = ["gauss"]#, "gauss"]
proposals = ["first_order"]
# step_sizes = [str(round(i, 3)) for i in np.linspace(1.0, 1.6, 61)]
# step_sizes = [str(round(i, 3)) for i in np.linspace(0.45, 0.55, 11)]
step_sizes = [str(round(i, 3)) for i in np.linspace(0.03, 0.05, 21)]


seeds = [f'seed_{i}' for i in range(10)]

# Define true parameter values for RMSE calculation and plotting
true_values = {
    'mean_rc_x_0': 0.75,
    'mean_rc_x_1': 1.2,
}

# Function to compute average runtime across seed directories
def compute_average_runtime(seed_dirs):
    fpath = os.path.join(seed_dirs[0].rsplit('/', 1)[0], 'average_runtime.csv')
    runtimes = []
    for seed_dir in seed_dirs:
        runtime_file = os.path.join(seed_dir, 'runtime_iterations.csv')
        df = pd.read_csv(runtime_file, header=None)
        runtimes.append(df.iloc[:,0].tolist()[-1])

    np.savetxt(fpath, np.array([np.mean(runtimes)]))

    return fpath

# Function to compute the mean across the same indices of multiple seed DataFrames
def compute_average_non_var(seed_dirs):
    fpath = os.path.join(seed_dirs[0].rsplit('/', 1)[0], 'avg_non_var_output.csv')
    dfs = []
    for seed_dir in seed_dirs:
        non_var_file = os.path.join(seed_dir, 'non_var_output.csv')
        df = pd.read_csv(non_var_file, index_col=0)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=1)
    mean_df = combined_df.groupby(level=0, axis=1).mean()
    mean_df.to_csv(fpath)

    return fpath

# Function to compute RMSE for parameters against true values and save to a file
def compute_rmse(avg_non_var_path, true_values):
    fpath = os.path.join(avg_non_var_path.rsplit('/', 1)[0], 'rmse.csv')
    avg_non_var_df = pd.read_csv(avg_non_var_path)
    rmses = []
    for param, true_value in true_values.items():
        last_value = avg_non_var_df[param].iloc[-1]
        rmse = np.sqrt(np.mean((last_value - true_value) ** 2))
        rmses.append({'param': param, 'rmse': rmse})
    rmse_df = pd.DataFrame(rmses)
    rmse_df.to_csv(fpath, index=False)


# Function to plot Dahlin convergence for parameters
def plot_dahlin_conv(avg_non_var_path, true_values):
    fpath = os.path.join(avg_non_var_path.rsplit('/', 1)[0], 'dahlin_conv.png')
    plt.figure()
    avg_non_var_df = pd.read_csv(avg_non_var_path)
    plt.plot(avg_non_var_df['mean_rc_x_0'], avg_non_var_df['mean_rc_x_1'], marker='o')
    for param, true_value in true_values.items():
        if 'mean_rc_x_0' in param:
            plt.axvline(x=true_value, color='r', linestyle='--', label='True mean_rc_x_0')
        elif 'mean_rc_x_1' in param:
            plt.axhline(y=true_value, color='b', linestyle='--', label='True mean_rc_x_1')

    plt.xlabel('mean_rc_x_0')
    plt.ylabel('mean_rc_x_1')
    plt.legend()
    plt.savefig(fpath)
    plt.close()


# Function to plot ESS for each proposal in an l_kernel directory
def plot_ess(l_kernel_dir, proposals):
    fpath = os.path.join(l_kernel_dir, 'ess.png')
    plt.figure()
    for proposal in proposals:
        avg_non_var_df = pd.read_csv(os.path.join(l_kernel_dir, proposal, 'avg_non_var_output.csv'))
        print(avg_non_var_df['ess'])
        plt.plot(avg_non_var_df.index, avg_non_var_df['ess'], label=proposal)
    plt.xlabel('Iterations')
    plt.ylabel('ESS')
    plt.title('Average ESS per Proposal')
    plt.legend()
    plt.savefig(fpath)
    plt.close()

# Function to plot parameter convergence for a given parameter
def plot_conv_param(l_kernel_dir, proposals, param_name, true_values):
    plt.figure()
    for proposal in proposals:
        avg_non_var_df = pd.read_csv(os.path.join(l_kernel_dir, proposal, 'avg_non_var_output.csv'))
        plt.plot(avg_non_var_df.index, avg_non_var_df[param_name], label=proposal)
    plt.axhline(y=true_values[param_name], color='r', linestyle='--', label='True Value')
    plt.xlabel('Iterations')
    plt.ylabel(param_name)
    plt.legend()
    plt.savefig(os.path.join(l_kernel_dir, f'conv_{param_name}.png'))
    plt.close()


# Main function to process model structure and apply computations and plotting
def process_model_structure(base_dir, models, l_kernels, proposals, step_sizes, seeds, true_values):
    #need to consider outer loops
    for model in models:
        for step_size in step_sizes:
            for l_kernel in l_kernels:
                l_kernel_dir = os.path.join(base_dir, model, step_size, l_kernel)
                for proposal in proposals:
                    paths = []
                    for seed in seeds:
                        paths.append(os.path.join(base_dir, model, step_size, l_kernel, proposal, seed))
                    runtime_path = compute_average_runtime(paths)
                    avg_non_var_path = compute_average_non_var(paths)
                    compute_rmse(avg_non_var_path, true_values)
                    plot_dahlin_conv(avg_non_var_path, true_values)

                plot_ess(l_kernel_dir, proposals)
                plot_conv_param(l_kernel_dir, proposals, 'mean_rc_x_0', true_values)
                plot_conv_param(l_kernel_dir, proposals, 'mean_rc_x_1', true_values)


process_model_structure(base_dir, models, l_kernels, proposals, step_sizes, seeds, true_values)
