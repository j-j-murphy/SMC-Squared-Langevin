import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Base directory and model components setup
base_dir = 'outputs'
models = ["lgssm_16_new_hess"]#, "lgssm_16_new_hess
l_kernels = ["gauss", "forwards-proposal"]#, "gauss"]
proposals = ["first_order"]
#step_sizes = [str(round(i, 3)) for i in np.linspace(1.0, 1.8, 9)]
# step_sizes = [str(round(i, 3)) for i in np.linspace(0.1, 1.5, 57)]
step_sizes = [str(round(i, 3)) for i in np.linspace(0.01, 0.1, 10)]
#step_sizes = ["0.03"]

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
    print(df.shape)
    combined_df = pd.concat(dfs, axis=1)
    print(combined_df.shape)
    mean_df = combined_df.groupby(level=0, axis=1).mean()
    mean_df.to_csv(fpath)

    return fpath

def compute_average_var(seed_dirs):
    fpath = os.path.join(seed_dirs[0].rsplit('/', 1)[0], 'avg_var_rc.csv')
    dfs = []
    for seed_dir in seed_dirs:
        non_var_file = os.path.join(seed_dir, 'var_rc.csv')
        df = pd.read_csv(non_var_file, header=None)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=1)
    #print(combined_df.shape)
    mean_df = combined_df.groupby(level=0, axis=1).mean()
    mean_df.to_csv(fpath)

    return fpath

# Function to compute RMSE for parameters against true values and save to a file
def compute_rmse(avg_non_var_path, true_values):
    proposal_path = avg_non_var_path.rsplit('/', 1)[0]
    fpath = os.path.join(proposal_path, 'rmse.csv')
    avg_non_var_df = pd.read_csv(avg_non_var_path)
    return_rmses = {"path": proposal_path}
    rmses = []
    total_rmse = 0
    for param, true_value in true_values.items():
        last_value = avg_non_var_df[param].iloc[-1]
        rmse = np.sqrt(np.mean((last_value - true_value) ** 2))
        rmses.append({'param': param, 'rmse': rmse})
        return_rmses[param] = rmse
        total_rmse += rmse

    return_rmses['avg_rmse'] = total_rmse / len(true_values) 

    rmse_df = pd.DataFrame(rmses)
    rmse_df.to_csv(fpath, index=False)

    return return_rmses



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
    plt.xlim(0.7, 0.8)
    plt.ylim(1.1, 1.4)
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

def plot_conv_error_bars(paths, output_path, param_name, var_col, true_values):
    plt.figure()
    for path in paths:
        avg_non_var_df = pd.read_csv(os.path.join(path, 'avg_non_var_output.csv'))
        avg_var_df = pd.read_csv(os.path.join(path, 'avg_var_rc.csv'))
        plt.errorbar(avg_non_var_df.index, avg_non_var_df[param_name], yerr=np.sqrt(avg_var_df[var_col]), label=path.rsplit('/', 1)[1])
    plt.axhline(y=true_values[param_name], color='r', linestyle='--', label='True Value')
    plt.xlabel('Iterations')
    plt.ylabel(param_name)
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def compare_rmses(model_dir, rmses):
    fpath = os.path.join(model_dir, 'rmses.csv')
    if os.path.exists(fpath):
        print("hwer")
        rmses_df = pd.read_csv(fpath)
        rmses_df = pd.concat([rmses_df, pd.DataFrame(rmses)])
        rmses_df.drop_duplicates(['path'], inplace=True)
    else:
        rmses_df = pd.DataFrame(rmses)

    rmses_df = rmses_df.sort_values(by='avg_rmse')
    rmses_df.to_csv(fpath, index=False)    

    # print(fpath)
    # rmses = sorted(rmses, key=lambda x: x['avg_rmse'])
    # rmses_df = pd.DataFrame(rmses)
    # rmses_df.to_csv(fpath, index=False)


print("here")
# Main function to process model structure and apply computations and plotting
def process_model_structure(base_dir, models, l_kernels, proposals, step_sizes, seeds, true_values):
    #need to consider outer loops
    for model in models:
        model_dir = os.path.join(base_dir, model)
        rmses = []
        for step_size in step_sizes:
            step_size_dir = os.path.join(base_dir, model, step_size)
            for l_kernel in l_kernels:
                l_kernel_dir = os.path.join(base_dir, model, step_size, l_kernel)
                for proposal in proposals:
                    paths = []
                    for seed in seeds:
                        paths.append(os.path.join(base_dir, model, step_size, l_kernel, proposal, seed))
                    runtime_path = compute_average_runtime(paths)
                    avg_non_var_path = compute_average_non_var(paths)
                    avg_var_path = compute_average_var(paths)
                    avg_rmse = compute_rmse(avg_non_var_path, true_values)
                    plot_dahlin_conv(avg_non_var_path, true_values)

                    rmses.append(avg_rmse)

                plot_ess(l_kernel_dir, proposals)
                plot_conv_param(l_kernel_dir, proposals, 'mean_rc_x_0', true_values)
                plot_conv_param(l_kernel_dir, proposals, 'mean_rc_x_1', true_values)
        
        # save rmse and remember to sort
        compare_rmses(model_dir, rmses)

# process_model_structure(base_dir, models, l_kernels, proposals, step_sizes, seeds, true_values)

#paths to proposal dir
#fpath to combined
output_path = 'outputs/test.png'
comparison_paths = ['outputs/lgssm_16_new_hess/0.06/forwards-proposal/first_order', 
                    'outputs/lgssm_16_diag_hess/0.4/gauss/rw']
plot_conv_error_bars(comparison_paths, output_path, 'mean_rc_x_0', '0', true_values)
output_path = 'outputs/test2.png'

plot_conv_error_bars(comparison_paths, output_path, 'mean_rc_x_1', '3', true_values)
