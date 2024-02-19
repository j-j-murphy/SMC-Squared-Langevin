import os
from datetime import datetime
import numpy as np
import pandas as pd
from mpi4py import MPI
from pathlib import Path


class smc_no_diagnostics:
    def __init__(self):
        self.fpath = ""
    def make_run_folder(self):
        pass
    def save_iter_info(self, iteration, x, constrained_x, logw, seed):
        pass
    def save_final_info(self, mean, mean_rc, var, var_rc, Neff, resampling_points, N):
        pass

class smc_diagnostics_final_output_only(smc_no_diagnostics):
    def __init__(self, model, proposal, l_kernel, step_size, seed):
        self.now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.fpath = Path(f"outputs/{model}",f"{proposal}",f"{l_kernel}",f"{step_size}", f"seed_{int(seed)}")
        self.fpath = Path(f"outputs/{model}", f"{step_size}", f"{l_kernel}", f"{proposal}", f"seed_{int(seed)}")


    def make_run_folder(self):
        self.fpath.mkdir(parents=True, exist_ok=True)

    def save_final_info(self, mean, mean_rc, var, var_rc, Neff, resampling_points, N, runtime_iterations):
        #
        ess = Neff/N
        mean_df = pd.DataFrame(mean, columns = ["mean_x_"+str(i) for i in range(mean.shape[1])])
        mean_rc_df = pd.DataFrame(mean_rc, columns = ["mean_rc_x_"+str(i) for i in range(mean_rc.shape[1])])
        resampling_points = np.pad(resampling_points, (0, len(ess)-len(resampling_points)))
        non_estim_df = pd.DataFrame({"neff": Neff, "ess":ess, "resampling_points":resampling_points})
        final_info_df = pd.concat([mean_df, mean_rc_df, non_estim_df], axis=1)
        final_info_df.to_csv(Path(self.fpath, "non_var_output.csv"))

        if MPI.COMM_WORLD.Get_rank() == 0:
            print(final_info_df)

        var = np.reshape(var, (var.shape[0], -1))
        var_rc = np.reshape(var_rc, (var_rc.shape[0], -1))

        np.savetxt(Path(self.fpath,"var.csv"), var, delimiter=",")
        np.savetxt(Path(self.fpath,"var_rc.csv"), var_rc, delimiter=",")
        np.savetxt(Path(self.fpath,"runtime_iterations.csv"), runtime_iterations, delimiter=",")

class smc_diagnostics(smc_diagnostics_final_output_only):
        
    def save_iter_info(self, iteration, x, constrained_x, logw, seed):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        #create iteration folder
        #rank=1
        iteration_fol = "iteration_{}".format(iteration)
        iteration_path = f"{self.fpath}/{iteration_fol}"
        # if rank == 0:
        #     os.mkdir(iteration_path) 
        x_df = pd.DataFrame(x, columns = ["x_"+str(i) for i in range(x.shape[1])])
        cons_x_df = pd.DataFrame(constrained_x, columns = ["cons_x_"+str(i) for i in range(constrained_x.shape[1])])
        logw_df = pd.DataFrame(logw, columns = ["logw_"+str(i) for i in range(logw.shape[1])])
        #save samples
        iter_info_df = pd.concat([x_df, cons_x_df, logw_df], axis=1)
        
        iter_info_df.to_csv(f"{iteration_path}/iter_info_{rank}.csv")