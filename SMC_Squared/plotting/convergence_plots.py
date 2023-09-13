import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from pathlib import Path

def get_estimates(model, get_ess=False, iteration=10):
    model_dir = Path(f"outputs/{model}")
    print(model_dir)
    rc_estimates = {}
    for means, variances, variances_urc, runtimes in zip(model_dir.rglob('non_var_output.csv'), model_dir.rglob('var_rc.csv'), model_dir.rglob('var.csv'), model_dir.rglob('runtime_iterations.csv')):
        #compare paths 
        particle_core_mean = "/".join(str(means).split("/")[2:5])
        particle_core_variance = "/".join(str(variances).split("/")[2:5])
        if particle_core_mean!=particle_core_variance:
            print(particle_core_mean)
            print(particle_core_variance)
            print("differing paths")
            break
        #extract means and filter for recycled
        non_var_outputs_df = pd.read_csv(means)[:iteration]
        means_rc = non_var_outputs_df.filter(like="mean_rc_x")
        means_urc = non_var_outputs_df.filter(like="mean_x")
        #extract variances
        var_estimates_2d = np.genfromtxt(variances_urc, delimiter=",")[:iteration]
        var_estimates_2d_rc = np.genfromtxt(variances, delimiter=",")[:iteration]
        if var_estimates_2d_rc.ndim > 1:  
            var_estimates_rc = np.reshape(var_estimates_2d_rc, (var_estimates_2d_rc.shape[0], int(np.sqrt(var_estimates_2d_rc.shape[1])), -1))
            var_estimates_urc = np.reshape(var_estimates_2d, (var_estimates_2d.shape[0], int(np.sqrt(var_estimates_2d.shape[1])), -1))
        else:
            var_estimates_rc = var_estimates_2d_rc
            var_estimates_urc = var_estimates_2d

        ess = non_var_outputs_df.filter(like="ess")[:iteration]
        neff = non_var_outputs_df.filter(like="neff")[:iteration]
        times = np.genfromtxt(runtimes, delimiter=",")[:iteration]
        rc_estimates[particle_core_mean] = {"means": means_rc, "means_urc": means_urc, "variances": var_estimates_rc, "variances_urc": var_estimates_urc, "ess": ess, "neff": neff, "times": times}
        
    print(f"The number of estimates is {len(rc_estimates)}")
    return rc_estimates

def avg_estimates(estimates):
    #finding set of particles
    particle_cores = list(set("/".join(key.split("/")[0:2]) for key in estimates.keys()))
    #print(particle_cores)
    estimate_averages={}
    #averaing across all seeds
    #for particle_core in particle_cores:
    estimate_averages = {
        particle_core: {"mean": pd.concat([value["means"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])]).mean(level=0),
                        "mean_urc": pd.concat([value["means_urc"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])]).mean(level=0),
                        "variance": np.mean([value["variances"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])], axis=0),
                        "variance_urc": np.mean([value["variances_urc"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])], axis=0),
                        "time": np.mean([value["times"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])], axis=0),
                        "ess": pd.concat([value["ess"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])]).mean(level=0),
                        "neff": pd.concat([value["neff"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])]).mean(level=0)}
        for particle_core in particle_cores   
    }
    #[print(value) for key, value in mean_estimates.items()]
        # if particle_core=="particles_256/cores_1":
        #     print(particle_core, [value for key, value in runtimes.items() if particle_core==("/").join(key.split("/")[0:2])])
    #print(runtime_averages)
    return estimate_averages

def compare_estimates(avg_mean_estimates):
    pass

def gather_estimates(models, label=["Forwards Proposal", "Near Optimal"], particles="1024"):
    #TODO: for 
    model_comparison = {}
    for i, model in enumerate(models):
        particle_core1 = {}
        for key, value in model.items():
            #print(key)
            if "cores_128" in key.split("/")[1]:
                #taking 1 core only
                particle_core1[key.split("/")[0].split("_")[1]] = value

        particle_core1 = dict(sorted(particle_core1.items(), key=lambda x: int(x[0])))
        
        if particles:
            particle_core1 = particle_core1[particles]

        model_comparison[label[i]] = particle_core1

    for model, model_estimates in model_comparison.items():
        #print(model, model_estimates["mean"])
        pass
    #print(model_comparison)
    return model_comparison

def plot_estimates(particle_core1, parameters, average_estimates, particles="1024", iteration=0):
    cmap = plt.get_cmap('hsv')
    #print(particle_core1.keys())
    #print(particle_core1[list(particle_core1.keys())[0]]["mean"])
    if len(parameters) != 1:
        fig, ax = plt.subplots(ncols=len(parameters))

        for d in range(len(parameters)):
            for i, particle in enumerate(particle_core1):
                ax[d].errorbar(x=np.arange(0, len(particle_core1[particle]["mean"].iloc[:, d])), 
                                y=particle_core1[particle]["mean"].iloc[:, d], 
                                yerr=np.sqrt(particle_core1[particle]["variance"][:, d, d]), 
                                color=cmap(i / len(particle_core1), alpha=0.5), 
                                label=particle)
                # ax[d].errorbar(x=np.arange(0, len(particle_core1[particle]["mean_urc"].iloc[:, d])), 
                #                 y=particle_core1[particle]["mean_urc"].iloc[:, d], 
                #                 yerr=np.sqrt(particle_core1[particle]["variance_urc"][:, d, d]), 
                #                 color=cmap((i+0.5) / len(particle_core1), alpha=0.5), 
                #                 label=f"{particle} unrecycled")
                ax[d].set_title(list(parameters.keys())[d])
                ax[d].plot(np.repeat(list(parameters.values())[d], len(particle_core1[particle]["mean"])), 'lime', linewidth=3.0,
                       linestyle='--')
                ax[d].set_xlabel('Iteration')
                ax[d].set_ylabel('E[$x$]')
                if iteration:
                    ax[d].axis(xmin=0, xmax=iteration)
            #ax[d].legend([particle_plts], list(particle_core1.keys()), title  = "Number of \n Particles")
        plt.legend(title  = "Number of Particles")
        #plt.legend()

    else:
        fig, ax = plt.subplots(ncols=len(parameters), squeeze=False)

        for d in range(len(parameters)):
            for i, particle in enumerate(particle_core1):
                print(particle)
                ax[d, 0].errorbar(x=np.arange(0, len(particle_core1[particle]["mean"].iloc[:, d])), 
                                y=particle_core1[particle]["mean"].iloc[:, d], 
                                yerr=np.sqrt(particle_core1[particle]["variance"]), 
                                color=cmap(i / len(particle_core1), alpha=0.2),
                                label=particle)
                ax[d, 0].errorbar(x=np.arange(0, len(particle_core1[particle]["mean_urc"].iloc[:, d])), 
                                y=particle_core1[particle]["mean_urc"].iloc[:, d], 
                                yerr=np.sqrt(particle_core1[particle]["variance_urc"]), 
                                color=cmap((i+0.5) / len(particle_core1), alpha=0.2),
                                label=f"{particle} unrecycled")
                ax[d, 0].set_title(list(parameters.keys())[d])
                ax[d, 0].plot(np.repeat(list(parameters.values())[d], len(particle_core1[particle]["mean"])), 'lime', linewidth=3.0,
                       linestyle='--')
                ax[d, 0].set_xlabel('Iteration')
                ax[d, 0].set_ylabel('E[$x$]')
                if iteration:
                    ax[d, 0].axis(xmin=0, xmax=iteration)
        plt.legend([particle_plt], [particle], title  = "Number of Particles")
    
    plt.tight_layout()
    
    
    if particles:
        plt.savefig(f"outputs/{model}/{model.split('.')[0]}_convergence_{particles}_particles.png")
    else:
        plt.savefig(f"outputs/{model}/{model.split('.')[0]}_convergence_{iteration}.png")

def get_mse(model_comparison, parameters, iteration=10):
    for model in model_comparison:
        estimates = model_comparison[model]["mean"].iloc[iteration].to_numpy()
        #print("final estimates", estimates)
        squared_errors = (estimates - np.array(list(parameters.values()))) ** 2
        #mse = np.mean(squared_errors)
        print(model, squared_errors)

def get_time(model_comparison, iteration=10):
    for model in model_comparison:
        times = model_comparison[model]["time"][iteration]
        print(model, times)


if __name__=="__main__":
    model_name = "SIR"
    #model="SIR_forwards-proposal"
    model=f"{model_name}_forwards-proposal"
    model2 = f"{model_name}_gauss"

    if model_name=="SIR":
        parameters = {"$\Theta$": 0.85,  
                    "$\gamma$": 0.2}
    elif model_name=="SEIR":
        parameters = {"$\\beta$": 0.9,
                        "$\gamma$": 0.08,
                        "$\delta$": 0.3}
    elif model_name=="SEIIR":
        parameters = {"$\\beta$": 0.9,
                        "$\gamma_1$": 0.08,
                        "$\gamma_2$": 0.1,
                        "$\delta$": 0.3}
    

    estimates = get_estimates(model)
    estimates2 = get_estimates(model2)
    # average_estimates = avg_estimates(estimates)
    # average_estimates2 = avg_estimates(estimates2)
    # #gather_estimates_dict = gather_estimates([average_estimates, average_estimates2], label=["Forwards Proposal", "Near Optimal"], particles="256")
    # #plot_estimates(gather_estimates_dict, parameters, avg_estimates, particles=None, iteration=50)
    
    # particles = ["1024"]
    # for particle in particles:
    #     print("particle", particle, "____")
    #     gather_estimates_dict = gather_estimates([average_estimates, average_estimates2], label=["Forwards Proposal", "Near Optimal"], particles=particle)
    #     mse = get_mse(gather_estimates_dict, parameters, iteration=9)
        #time = get_time(gather_estimates_dict, iteration=10)
    #print(avg_estimates)
    # mean_estimates = get_mean_estimates(model)
    # average_mean_estimates = avg_mean_estimates(mean_estimates)
    # plot_mean_estimates(model, parameters, average_mean_estimates)
    
    #model="SIR_gauss"

    mean_estimates_seeds = np.array([estimates2[seed]["means"].iloc[-1].to_numpy() for seed in estimates2.keys() if "particles_1024/cores_32" in seed])
    mean_estimate = np.mean(mean_estimates_seeds, axis=0)
    mses = np.zeros((10,1))
    squared_errors = np.zeros((10,2))

    for i in range(len(mean_estimates_seeds)):
        squared_errors[i] = (mean_estimates_seeds[i] - np.array(list(parameters.values()))) ** 2
        mses[i] = np.mean(squared_errors[i])


    #print(mean_estimates_seeds)
    #print(average_estimates2)
    print("SMC")
    print("Mean of Parameter Values", mean_estimate)
    print("Mean of Parameter of MSEs", np.mean(squared_errors, axis=0))
    print("Mean of combined parameter MSEs", np.mean(mses, axis=0))
    print("Median of Parameter of MSEs", np.median(squared_errors, axis=0))
    print("Median of combined parameter MSEs", np.median(mses, axis=0))
    print(np.argmin(mses))

    seeds = range(0,10)

    for seed in seeds:
        mean_estimates_best_seed = np.array([estimates2[run]["means"].to_numpy() for run in estimates2.keys() if f"particles_1024/cores_32/seed_{seed}" in run])
        print(mean_estimates_best_seed)

        with open(f"outputs/{model2}/particles_1024/cores_32/seed_{seed}/mean_estimates.pickle", 'wb') as handle:
            pickle.dump(mean_estimates_best_seed, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fig, ax = plt.subplots(ncols=len(parameters))
        for d in range(len(parameters)):
            ax[d].plot(mean_estimates_best_seed[0][:, d])
            ax[d].plot(np.repeat(list(parameters.values())[d], len(mean_estimates_best_seed)), 'lime', linewidth=3.0,
                   linestyle='--')
            ax[d].set_title(f"{list(parameters.keys())[d]}")
            ax[d].set_xlabel('Iteration')
            ax[d].set_ylabel('E[$x$]')
        
        print("here")

        plt.savefig(f"outputs/{model2}/particles_1024/cores_32/seed_{seed}/mean_estimates_smc.png")
        #print(mean_estimates_best_seed)
    