# HESS-MC2: SEQUENTIAL MONTE CARLO SQUARED USING HESSIAN INFORMATION AND SECOND ORDER PROPOSALS

## Models
```target_pf.py``` provides a general-purpose ```logpdf``` method to return the log posterior $\pi(\theta)$, gradient of the log posterior $\nabla\pi(\theta)$ and Hessian of the log posterior $\nabla^2\pi(\theta)$. Particle filters such as ```sir_PF``` in ```sir.py``` inherit from this object while specifying the dynamics and weight update in a ```run_particleFilter()``` method which returns the loglikelihood $p(\mathbf{y}_{1:t}|\theta)$

## Running Parameter Estimation 
Parameter estimation can be performed by running for example ```SMC.sh param_est_sir.py``` on systems with the SLURM workload manager. On individual workstations you can run ```mpiexec -n $number_of_cores python param_est_sir.py``` 

## Results Generation
Results are generated using '''plotting.py``` and ```boxplots.py```


## Requirements
To install required packages run ```pip install -r requirements.txt```

## References
The distributed SMC$^2$ framework is described in:
Rosato C, Varsi A, Murphy J, Maskell S. An O (log 2 N) SMC 2 Algorithm on Distributed Memory with an Approx. Optimal L-Kernel. In2023 IEEE Symposium Sensor Data Fusion and International Conference on Multisensor Fusion and Integration (SDF-MFI) 2023 Nov 27 (pp. 1-8). IEEE.
 
The method for differentiating the particle filter is outlined in:
Rosato C, Devlin L, Beraud V, Horridge P, Schön TB, Maskell S. Efficient learning of the parameters of non-linear models using differentiable resampling in particle filters. IEEE Transactions on Signal Processing. 2022 Jul 1;70:3676-92.
 
The use of first-order gradients in Langevin proposals in SMC$^2$ is described in:
Rosato C, Murphy J, Varsi A, Horridge P, Maskell S. Enhanced SMC 2: Leveraging Gradient Information from Differentiable Particle Filters Within Langevin Proposals. In2024 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI) 2024 Sep 4 (pp. 1-8). IEEE.
