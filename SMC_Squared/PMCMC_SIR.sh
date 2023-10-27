#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job.
#SBATCH --export=ALL
# Define job name

# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o logs/%x.out
#SBATCH --exclusive

# Request the partition
#SBATCH -p bighyp
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -J PMCMC_SIR_new

#SBATCH -c 1
# This asks for 10 minutes of time.
#SBATCH -t 72:00:00
# Specify memory per core
##SBATCH --mem-per-cpu=8000M
# Insert your own username to get e-mail notifications
##SBATCH --mail-user=<username>@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL
#
# Edit to match your own executable and stdin (/dev/null if no stdin)
# Note the assumption about where these reside in your home directory! 

# Load relevant modules
module load apps/anaconda3/5.2.0
module load mpi/openmpi/1.10.7/gcc-5.5.0
source activate mpi4py_env

# Activate your own virtual environment in which mpipy is installed
# If you haven't set up such virtual environment, you could create one firstly:
#  conda create -n mpi4py_env mpi4py
# source activate mpi4py_env
#conda activate mpi4py_env (for new version)
echo "mpiexec=`which mpiexec`"
echo "mpirun=`which mpirun`"

#
# Should not need to edit below this line
#
echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo Executable file:                              
echo MPI parallel job.                                  
echo -------------  
echo Job output begins                                           
echo -----------------                                           
echo

hostname

echo "Print the following environmetal variables:"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"




echo "Running parallel job:"

# If you use all of the slots specified in the -pe line above, you do not need
# to specify how many MPI processes to use - that is the default
# the ret flag is the return code, so you can spot easily if your code failed.
python pmcmc_SIR.py

ret=$?

# Deactivate current active environment
# source deactivate mpi4py_env
# conda deactivate (for new version)


echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
exit $ret
