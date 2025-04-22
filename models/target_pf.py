import numpy as np
import sys
sys.path.append('..')  # noqa

import torch
from torch.autograd import grad

class Target_PF:   
    def __init__(self, prior_pdf, y, proposal_, P=1024):
        self.y = y
        self.proposal_ = proposal_
        self.P = P
        self.prior_pdf = prior_pdf
    
    def logpdf(self, thetas, rngs):
        try:
            thetas_ = torch.tensor(thetas, requires_grad=True)
            LL = self.run_particleFilter(thetas_, rngs)
            LL_=LL.detach().numpy() + self.prior_pdf(thetas)
            if self.proposal_ == 'rw':
                test1 = np.full((len(thetas), len(thetas)), -np.inf)     
                test = np.full(len(thetas), -np.inf)

            elif self.proposal_ == 'first_order':
                first_derivative = grad(LL, thetas_, create_graph=True)[0]
                first_derivative  = first_derivative.detach().numpy() #+ grad_prior_mu_first.detach().numpy()
                test = first_derivative
                test1 = np.full((len(thetas), len(thetas)), -np.inf)
                
            elif self.proposal_ == 'second_order':
                first_derivative = grad(LL, thetas_, create_graph=True)[0]
                second_derivative = [torch.autograd.grad(first_derivative[i], thetas_, create_graph=True)[0].detach().numpy() for i in range(len(thetas_))]
                second_derivative = np.stack(second_derivative)

                for i in range(len(thetas_)):
                    for j in range(len(thetas_)):
                        if i != j:
                            second_derivative[i][j] = 0

                            
                test = first_derivative.detach().numpy() 
                test1 = second_derivative #+ grad_prior_mu_second.detach().numpy()

            # set nans in test and test1 to -inf
            test[np.isnan(test)] = -np.inf
            test1[np.isnan(test1)] = -np.inf
        
        except:
            LL_ = -np.inf
            test = np.full(len(thetas), -np.inf)
            test1 = np.full((len(thetas), len(thetas)), -np.inf)

        return(LL_, test, test1)