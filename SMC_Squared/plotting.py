import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from pathlib import Path

def get_estimates(model):
    model_dir = Path(f"outputs/{model}")
    rc_estimates = {}
    for means, variances in zip(model_dir.rglob('non_var_output.csv'), model_dir.rglob('var_rc.csv')):
        #compare paths 
        particle_core_mean = "/".join(str(means).split("/")[2:5])
        particle_core_variance = "/".join(str(variances).split("/")[2:5])
        if particle_core_mean!=particle_core_variance:
            print(particle_core_mean)
            print(particle_core_variance)
            print("differing paths")
            break
        #extract means and filter for recycled
        non_var_outputs_df = pd.read_csv(means)
        means_rc = non_var_outputs_df.filter(like="mean_rc_x")
        #extract variances
        var_estimates_2d_rc = np.genfromtxt(variances, delimiter=",")
        if var_estimates_2d_rc.ndim > 1:  
            var_estimates_rc = np.reshape(var_estimates_2d_rc, (var_estimates_2d_rc.shape[0], int(np.sqrt(var_estimates_2d_rc.shape[1])), -1))
        else:
            var_estimates_rc = var_estimates_2d_rc

        rc_estimates[particle_core_mean] = {"means": means_rc, "variances": var_estimates_rc}

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
                        "variance": np.mean([value["variances"] for key, value in estimates.items() if particle_core==("/").join(key.split("/")[0:2])], axis=0)}
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
    model_comparison = {}
    for i, model in enumerate(models):
        particle_core1 = {}
        for key, value in model.items():
            print(key)
            if key.split("/")[1]=="cores_1":
                #taking 1 core only
                particle_core1[key.split("/")[0].split("_")[1]] = value

        particle_core1 = dict(sorted(particle_core1.items(), key=lambda x: int(x[0])))
        if particles:
            particle_core1 = particle_core1[particles]

        model_comparison[label[i]] = particle_core1
    print(model_comparison)
    return model_comparison

def plot_estimates(particle_core1, parameters, average_estimates, particles="1024"):
    cmap = plt.get_cmap('viridis')
    #print(particle_core1.keys())
    #print(particle_core1[list(particle_core1.keys())[0]]["mean"])
    if len(parameters) != 1:
        fig, ax = plt.subplots(ncols=len(parameters))

        for d in range(len(parameters)):
            for i, particle in enumerate(particle_core1):
                particle_plt = ax[d].errorbar(x=np.arange(0, len(particle_core1[particle]["mean"].iloc[:, d])), 
                                y=particle_core1[particle]["mean"].iloc[:, d], 
                                yerr=np.sqrt(particle_core1[particle]["variance"][:, d, d]), 
                                color=cmap(i / len(particle_core1), alpha=0.5), 
                                label=particle)
                ax[d].set_title(list(parameters.keys())[d])
                ax[d].plot(np.repeat(list(parameters.values())[d], len(particle_core1[particle]["mean"])), 'lime', linewidth=3.0,
                       linestyle='--')
                ax[d].set_xlabel('Iteration')
                ax[d].set_ylabel('E[$x$]')
            #ax[d].legend([particle_plts], list(particle_core1.keys()), title  = "Number of \n Particles")
            ax[d].legend(title  = "Number of \n Particles")
        #plt.legend()

    else:
        fig, ax = plt.subplots(ncols=len(parameters), squeeze=False)

        for d in range(len(parameters)):
            for i, particle in enumerate(particle_core1):
                print(particle)
                particle_plt = ax[d, 0].errorbar(x=np.arange(0, len(particle_core1[particle]["mean"].iloc[:, d])), 
                                y=particle_core1[particle]["mean"].iloc[:, d], 
                                yerr=np.sqrt(particle_core1[particle]["variance"]), 
                                color=cmap(i / len(particle_core1), alpha=0.5),
                                label=particle)
                ax[d, 0].set_title(list(parameters.keys())[d])
                ax[d, 0].plot(np.repeat(list(parameters.values())[d], len(particle_core1[particle]["mean"])), 'lime', linewidth=3.0,
                       linestyle='--')
                ax[d, 0].set_xlabel('Iteration')
                ax[d, 0].set_ylabel('E[$x$]')
                ax[d, 0].legend([particle_plt], [particle], title  = "Number of \n Particles")
    
    plt.tight_layout()
    
    
    if particles:
        plt.savefig(f"outputs/{model}/{model.split('.')[0]}_convergence_{particles}_particles.png")
    else:
        plt.savefig(f"outputs/{model}/{model.split('.')[0]}_convergence.png")

if __name__=="__main__":
    #model="SIR_forwards-proposal"
    model="SIR_forwards-proposal"
    model2 = "SIR_gauss"
    parameters = {"$\Theta$": 0.65,  
                    "$\gamma$": 0.3}

    estimates = get_estimates(model)
    estimates2 = get_estimates(model2)
    average_estimates = avg_estimates(estimates)
    average_estimates2 = avg_estimates(estimates2)
    gather_estimates = gather_estimates([average_estimates, average_estimates2], label=["Forwards Proposal", "Near Optimal"], particles="256")
    plot_estimates(gather_estimates, parameters, avg_estimates, particles=None)
    #print(avg_estimates)
    # mean_estimates = get_mean_estimates(model)
    # average_mean_estimates = avg_mean_estimates(mean_estimates)
    # plot_mean_estimates(model, parameters, average_mean_estimates)
    
    #model="SIR_gauss"
    