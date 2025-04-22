import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# Base directory and model components setup
base_dir = 'outputs'
models = ["sir_N_32_K_15_D_2_P_500"]
# models = ["svm_N_32_K_15_D_3_P_500"]
# models = ["lgssm_N_32_K_15_D_3_P_500_fix_v"]
# models = ["earthquake_N_32_K_15_D_3_P_500"]
l_kernels = ["forwards-proposal"]

seed_nums = range(1,50)
seeds = [f"seed_{seed_num}" for seed_num in seed_nums]
# other_seeds = range(1,11)
# other_seeds = [f"seed_{seed_num}" for seed_num in other_seeds]
# seeds = seeds + other_seeds

# sec_order_ss = np.linspace(0.7, 1.7, 21)
# first_order_ss = np.linspace(0.02, 0.03, 21)
# rw_ss = np.linspace(0.1, 0.9, 17)
### LGSS ###
rw_ss = [0.1] 
first_order_ss = [0.1]
sec_order_ss = [1.15]
###
# ### SIR ###
rw_ss = [0.15] 
first_order_ss = [0.0089]
sec_order_ss = [2.5]
# ###
sec_order_ss = [str(round(i,2)) for i in sec_order_ss]
first_order_ss = [str(round(i,4)) for i in first_order_ss]
rw_ss = [str(round(i,3)) for i in rw_ss]
# sec_order_ss = ["1.5"]
# first_order_ss = ["0.085"]
# rw_ss = ["0.175"]
# sec_order_ss = np.linspace(0.8, 1.8, 11)
# sec_order_ss = [str(round(i,2)) for i in sec_order_ss]
props = {'rw': (rw_ss, seeds), 'first_order': (first_order_ss, seeds), 'second_order': (sec_order_ss, seeds),  }#, 'rw': (rw_ss, seeds)}#, }#'first_order': (first_order_ss, seeds)}#,

# Define true parameter values for RMSE calculation and plotting
if models[0] == "svm_N_32_K_15_D_3_P_500":
    true_values = {
        'mean_rc_x_0': 0.9,
        'mean_rc_x_1': 0.15,
        'mean_rc_x_2': 0.15
    }
elif models[0] == "earthquake_N_32_K_15_D_3_P_500":
    true_values = {
        'mean_rc_x_0': 0.88,
        'mean_rc_x_1': 0.15,
        'mean_rc_x_2': 16.58 # 17.65
    }
elif models[0] == "sir_N_32_K_15_D_2_P_500":
    true_values = {
        'mean_rc_x_0': 0.6,
        'mean_rc_x_1': 0.3
    }
elif models[0] == "lgssm_N_32_K_15_D_3_P_500_fix_v":
    true_values = {
        'mean_rc_x_0': 0.75,
        'mean_rc_x_1': 1.0,
        'mean_rc_x_2': 1.0
    }

# 0.88, 0.15, 17.65

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
    nan_paths = []
    for seed_dir in seed_dirs:
        non_var_file = os.path.join(seed_dir, 'non_var_output.csv')
        df = pd.read_csv(non_var_file, index_col=0)
        # if any nans in df, print seed_dir
        if df.isnull().values.any():
            nan_paths.append(seed_dir.split('/')[-1].split('_')[1])
        else:
            dfs.append(df)
    try:
        combined_df = pd.concat(dfs, axis=1)
    except:
        print("seed_dirs", seed_dirs)
    combined_df = pd.concat(dfs, axis=1)
    mean_df = combined_df.groupby(level=0, axis=1).mean()
    mean_df.to_csv(fpath)
    if len(nan_paths) > 0:
        ss, l_kernels, prop =seed_dirs[0].split('/')[-4: -1]
        print(f"{prop}-{ss} have nans in {nan_paths}")

    return fpath

def compute_avg_var(seed_dirs):
    rcs = ["var.csv", "var_rc.csv"]
    fpaths = []
    for rc in rcs:
        fpath = os.path.join(seed_dirs[0].rsplit('/', 1)[0], f'avg_{rc}')
        dfs = []
        for seed_dir in seed_dirs:
            var_file = os.path.join(seed_dir, rc)
            var_arr = pd.read_csv(var_file, header=None).to_numpy()
            # add var_arr to dfs
            dfs.append(pd.DataFrame(var_arr))
        var_arr = np.array(dfs)
        avg_var_arr = np.mean(var_arr, axis=0)
        avg_var_df = pd.DataFrame(avg_var_arr).to_csv(fpath)
        fpaths.append(fpath)
    
    return fpaths

# Function to compute RMSE for parameters against true values and save to a file
def compute_rmse(avg_non_var_paths, true_values):
    rcs = ["non_rc", "rc"]
    num_params = len(true_values)

    name = []
    rmse_rc = []
    # create df with name, rmse for each parameter and avg param
    means_rc = ["mean_rc_x_" + str(i) for i in range(num_params)]
    means = ["mean_x_" + str(i) for i in range(num_params)]
    df_rmse = pd.DataFrame(columns=['prop_name',  'avg_rmse', 'avg_rmse_rc'])
    # add to df_rmse only if prop_name not already in df_rmse
    if len(avg_non_var_paths) > 3:
        fpath = os.path.join("/".join(avg_non_var_paths[0].split("/")[:2]), 'rmse.csv')
    else:
        fpath = os.path.join("/".join(avg_non_var_paths[0].split("/")[:2]), 'rmse._select.csv')
    for rc in rcs:
        for avg_non_var_path in avg_non_var_paths:
            prop_name = str("/".join(avg_non_var_path.split("/")[2:5]))
            # add to df_rmse only if prop_name not already in df_rmse
            if prop_name not in df_rmse['prop_name'].values:
                df_rmse = pd.concat([df_rmse, pd.DataFrame([{'prop_name': prop_name}])], ignore_index=True)

            avg_non_var_df = pd.read_csv(avg_non_var_path)
            rmses_prop = []
            for i in range(num_params):
                if rc == "rc":
                    param = "mean_rc_x_" + str(i)
                    avg_rmse_col = "avg_rmse_rc"
                else:
                    param = "mean_x_" + str(i)
                    avg_rmse_col = "avg_rmse"

                true_value = true_values[list(true_values.keys())[i]]
            
                rmse = np.mean((avg_non_var_df[param] - true_value) ** 2)
                # write rmse to df_rmse
                df_rmse.loc[df_rmse['prop_name'] == prop_name, param] = rmse
                rmses_prop.append(rmse)

            avg_rmse = np.mean(rmses_prop)
            df_rmse.loc[df_rmse['prop_name'] == prop_name, avg_rmse_col] = avg_rmse

    # sort df_rmse by avg_rmse
    df_rmse = df_rmse.sort_values(by='avg_rmse_rc')
    print(df_rmse)   
    df_rmse.to_csv(fpath, index=False)
    

# Function to plot Dahlin convergence for parameters
def plot_dahlin_conv(avg_non_var_paths, true_values, paramss=[r"$\mu$", r"$\phi$", r"$\sigma$"], markers=['*', 'o', 'x']):
    # df = pd.read_csv(avg_non_var_path)
    paramss = [r"$\mu$", r"$\phi$", r"$\sigma$"]
    num_params = len(true_values)
    rcs = ["non_rc", "rc"]

    for rc in rcs:
        fig, axs = plt.subplots(1, num_params, figsize=(15, 5))
        for avg_non_var_path in avg_non_var_paths:
            df = pd.read_csv(avg_non_var_path)
        # loop through true values and plot, set up subplot comparing all 3 values
    
            for i in range(num_params):
                j = i + 1
                # if j greater than 2, set j to 0
                if j > num_params - 1:
                    j = 0
                if rc == "rc":
                    param_i = "mean_rc_x_" + str(i)
                    param_j = "mean_rc_x_" + str(j)
                else:
                    param_i = "mean_x_" + str(i)
                    param_j = "mean_x_" + str(j)


                axs[i].plot(df[param_i], df[param_j], label=avg_non_var_path.split('/')[-2], marker=markers[avg_non_var_paths.index(avg_non_var_path)])
                axs[i].axvline(x=true_values[list(true_values.keys())[i]], color='black')
                axs[i].axhline(y=true_values[list(true_values.keys())[j]], color='black')
                axs[i].set_xlabel(r"{}".format(paramss[i]), size=25)
                axs[i].set_ylabel(r"{}".format(paramss[j]), size=25)
                # axs[i].set_title(r"LGSSM: $\theta$= [{}, {}]".format(paramss_title[i], paramss_title[j]), size=25)
                
                axs[i].grid()
        
        axs[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join("/".join(avg_non_var_path.split("/")[:2]), f'dahlin_{rc}.png'))
        plt.close()

# Function to plot parameter convergence for a given parameter
def plot_conv_param(avg_non_var_paths, avg_var_paths, true_values):
    plt.figure()
    # avg var paths rc is second element of each list in avg_var_paths
    avg_var_paths_non_rcs = [avg_var_path[0] for avg_var_path in avg_var_paths]
    avg_var_paths_rcs = [avg_var_path[1] for avg_var_path in avg_var_paths]
    # get number of parameters
    num_params = len(true_values)
    rcs = ["non_rc", "rc"]

    for rc in rcs:
        # set up subplot
        fig, axs = plt.subplots(num_params, 1, figsize=(10, 10))
        for (avg_non_var_path, avg_var_paths_non_rc, avg_var_paths_rc) in zip(avg_non_var_paths, avg_var_paths_non_rcs, avg_var_paths_rcs):
            avg_non_var_df = pd.read_csv(avg_non_var_path)
            # load avg_var_path non rc as numpy from string
            avg_var_paths_non_rc = pd.read_csv(avg_var_paths_non_rc, index_col=0).to_numpy(dtype=np.float64)
            avg_var_paths_rc = pd.read_csv(avg_var_paths_rc, index_col=0).to_numpy(dtype=np.float64)

            for i in range(num_params):
                var_col = num_params * i + i 
                if rc == "rc":
                    param_name = "mean_rc_x_" + str(i)
                    vars = avg_var_paths_rc[:, var_col]
                    # var col is i squared -1 unless i is 0
                else:
                    param_name = "mean_x_" + str(i)
                    vars = avg_var_paths_non_rc[:, var_col]
                means = avg_non_var_df[param_name].to_numpy()

                vars = np.sqrt(vars)

                
                axs[i].fill_between(avg_non_var_df.index, means - vars, means + vars, alpha=0.5)
                axs[i].plot(avg_non_var_df.index, means, label=avg_non_var_path.split('/')[-2])
                axs[i].axhline(y=true_values[list(true_values.keys())[i]], color='r', linestyle='--')
                axs[i].set_xlabel('Iterations')
                axs[i].set_ylabel(param_name)
        
        axs[0].legend()
        plt.savefig(os.path.join("/".join(avg_non_var_path.split("/")[:2]), f'convergence_{rc}.png'))
        plt.close()


def generate_sir_results_table(avg_non_var_paths, base_dir='outputs'):
    latex_code = r"""\begin{table}[ht]
\centering
\caption{SIR Results}
\begin{tabular}{cccccc}
\hline 
\hline 
\textbf{Proposal} & $\epsilon$ & $\mathbf{E[\beta]}$ & $\mathbf{E[\gamma]}$ & \textbf{MSE} & \textbf{ESS}\\
\hline 
"""
    for avg_non_var_path in avg_non_var_paths:
        model_rmse_path = os.path.join(base_dir, avg_non_var_path.split('/')[-5], 'rmse.csv')
        df = pd.read_csv(avg_non_var_path)
        
        # Get proposal details
        proposal_name = avg_non_var_path.split('/')[-2]
        step_size = avg_non_var_path.split('/')[-4]
        row = df.iloc[-1]  # take last row
        average_all_rows = df.mean()
        
        # Get parameter estimates
        e_beta = f"{row['mean_rc_x_0']:.3f}"
        e_gamma = f"{row['mean_rc_x_1']:.3f}"
        ess = f"{average_all_rows['ess']:.3f}"
        
        # Get MSE
        mse_df = pd.read_csv(model_rmse_path)
        prop_mse_name = f"{step_size}/forwards-proposal/{proposal_name}"
        mse = mse_df.loc[mse_df['prop_name'] == prop_mse_name, 'avg_rmse_rc'].values[0]
        
        # Format MSE
        if mse < 0.001:
            mse_str = f"${mse:.2e}".replace('e-0', ' \\times 10^{-') + '}$'
        else:
            mse_str = f"{mse:.3f}"
        
        # Format proposal names
        proposal_name = proposal_name.replace('first_order', 'FO').replace('second_order', 'SO').replace('rw', 'RW')
        
        # Add row to table with epsilon
        latex_code += f"{proposal_name} & ${step_size}$ & ${e_beta}$ & ${e_gamma}$ & {mse_str} & ${ess}$\\\\\n"
    
    # Close the table
    latex_code += r"""\hline \hline
\end{tabular}
\label{Table:SIR_result}
\end{table}
"""
    print(latex_code)


def generate_lgssm_results_table(avg_non_var_paths, base_dir='outputs'):
    latex_code = r"""\begin{table}[ht]
\caption{LGSS Results}
\renewcommand{\arraystretch}{1.2}
\centering
\begin{tabular}{ccccccc}
\hline 
\hline 
\textbf{Proposal} & $\epsilon$ & $\mathbf{E[\mu]}$ & $\mathbf{E[\phi]}$ & $\mathbf{E[\sigma]}$ & \textbf{MSE} & \textbf{ESS}\\
\hline 
"""
    
    for avg_non_var_path in avg_non_var_paths:
        model_rmse_path = os.path.join(base_dir, avg_non_var_path.split('/')[-5], 'rmse.csv')
        df = pd.read_csv(avg_non_var_path)
        
        # Get proposal details
        proposal_name = avg_non_var_path.split('/')[-2]
        step_size = avg_non_var_path.split('/')[-4]  # This is our epsilon
        row = df.iloc[-1]  # take last row
        average_all_rows = df.mean()
        
        # Get parameter estimates
        e_mu = f"{row['mean_rc_x_0']:.3f}"
        e_phi = f"{row['mean_rc_x_1']:.3f}"
        e_sigma = f"{row['mean_rc_x_2']:.3f}"
        ess = f"{average_all_rows['ess']:.3f}"
        
        # Get MSE
        mse_df = pd.read_csv(model_rmse_path)
        prop_mse_name = f"{step_size}/forwards-proposal/{proposal_name}"
        mse = mse_df.loc[mse_df['prop_name'] == prop_mse_name, 'avg_rmse_rc'].values[0]
        
        # Format MSE
        if mse < 0.001:
            mse_str = f"${mse:.1e}".replace('e-0', ' \\times 10^{-') + '}$'
        else:
            mse_str = f"{mse:.3f}"
        
        # Format proposal names
        proposal_name = proposal_name.replace('first_order', 'First-order').replace('second_order', 'Second-order').replace('rw', 'RW')
        
        # Add row to table with step size (epsilon)
        latex_code += f"{proposal_name} & ${step_size}$ & ${e_mu}$ & ${e_phi}$ & ${e_sigma}$ & {mse_str} & ${ess}$\\\\\n"
    
    # Close the table
    latex_code += r"""\hline \hline
\end{tabular}
\label{Table:LGSSM_results}
\end{table}
"""
    print(latex_code)


def process_model_structure2(base_dir, models, l_kernels, props, true_values):
    #need to consider outer loops
    avg_non_var_paths = []
    avg_var_paths = []
    for model in models:
        for proposal, (step_sizes, seeds) in props.items():
            # print(proposal, step_sizes, seeds)
            for step_size in step_sizes:
                for l_kernel in l_kernels:
                    l_kernel_dir = os.path.join(base_dir, model, step_size, l_kernel)
                        # for proposal in proposals:
                    paths = []
                    for seed in seeds:
                        paths.append(os.path.join(base_dir, model, step_size, l_kernel, proposal, seed))

                    # averaging runtime, means and variances
                    runtime_path = compute_average_runtime(paths)
                    avg_non_var_path = compute_average_non_var(paths)
                    var_paths = compute_avg_var(paths)

                    # append mean and var for convergence plots
                    avg_non_var_paths.append(avg_non_var_path)
                    avg_var_paths.append(var_paths)

        
        compute_rmse(avg_non_var_paths, true_values)
        if model == "sir_N_32_K_15_D_2_P_500":
            generate_sir_results_table(avg_non_var_paths)
        elif model == "lgssm_N_32_K_15_D_3_P_500_fix_v":
            generate_lgssm_results_table(avg_non_var_paths)
        
        plot_dahlin_conv(avg_non_var_paths, true_values)
        plot_conv_param(avg_non_var_paths, avg_var_paths, true_values)
        #plot dahlin

process_model_structure2(base_dir, models, l_kernels, props, true_values)