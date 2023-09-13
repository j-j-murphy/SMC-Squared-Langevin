# SMC_MC_approx_optL

## Version 1
- First commit of code from previous optL paper

## Version 2
- Added Monte-Carlo approximation of the optimal L-kernel (```SMC_OPT_MC```)
- Removed option for Gibbs-style sampling
- Removed option for manual L-kernel choice
- Added verbose option to ```SMC_BASE``` class

## Version 3
- Automatic testing in Github
- Introduced multi-modal example
- New proposals subdirectory where pre-written proposals classes will sit. All proposals will be based on Q_Base.
- SMC initializer now handles the incoming proposal type, including user-defined assuming it is based on Q_Base.
- D_dim_gauss.py uses a random walk proposal, while D_dim_gauss_mixture.py, uses a user-defined proposal.
- Changed test file to match new SMC class call.
- Import templates for targets, prior and proposal from SMC_TEMPLATES
- Fixed issue that was causing biased variance estiamtes when using recycled estimates 
-```SMC_HMC_BASE`` allows a hmc proposal to be utilised. It requires the step-size and number of steps for the Leapfrog integrator
- Cov parameter in ```SMC_HMC_BASE`` allows the covariance of the Gaussian which velocity is drawn from to be changed.
- Included requirements script which makes sure autodiff is available.

## Version 4
- Introduced separate file of recycling schemes

## Version 5
- Introduced automated testing for SMC_HMC_BASE.py
- Users to sample from Stan models via the PyBindStan interface
- Removed Github workflows (all tests now run locally instead)

## Development
- More efficient calculation in ESS recycling scheme (issue #17)

## Requirements
- To install required packages run ```pip install -r requirements.txt```



