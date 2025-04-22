import numpy as np
import sys
sys.path.append('..')  # noqa

import torch
from torch.autograd import grad

from models.target_pf import Target_PF

def generateData(Npop, t_length, beta, gamma, I_initial):
    S_ = torch.zeros(t_length)
    I_ = torch.zeros(t_length)
    R_ = torch.zeros(t_length)
    infected_ = torch.zeros(t_length)
    
    S = (Npop-I_initial)
    I=I_initial
    R=0
    
    I_[0] = I
    S_[0] = S
    R_[0] = 0 
    infected_[0] = 1.
    
    for t in range(1, t_length):
        dSdt = -beta * I * S/Npop
        dIdt = beta * I * S/Npop- gamma * I
        dRdt = gamma * I
        S = S+dSdt
        I = I+dIdt
        R = R+dRdt
        infected_[t] = torch.distributions.poisson.Poisson(I).sample()

        I_[t] = I
        S_[t] = S
        
    return(infected_, I_, S_)

class sir_PF(Target_PF):
    def __init__(self, prior_pdf, y, Npop, proposal_, P):
        super().__init__(prior_pdf, y, proposal_, P)
        self.Npop = Npop

    def run_particleFilter(self, thetas, rngs):
        T = len(self.y)
        P = self.P
        beta, gamma = thetas
        
        Npop = self.Npop
        S = torch.zeros((T, P))
        I = torch.zeros((T, P))
        R = torch.zeros((T, P))
        lw = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        I[0] = torch.full((1, P),1.)[0] #torch.full((1, P),1.)[0] #torch.ones(P)

        S[0] = Npop- I[0]
        I[0] = I[0]
        R[0] = torch.zeros(P)
        lw[:] = -torch.log(torch.ones(1,1)*P)
        noise_b = rngs.torchNormalrsample(torch.tensor([T, P]), mu=0, sigma=0.5)#rngs.torchNormalrsample(0, 0.5).rsample(torch.tensor([T, P]))
        noise_g = rngs.torchNormalrsample(torch.tensor([T, P]), mu=0, sigma=0.5)#rngs.torchNormalrsample(0, 0.5).rsample(torch.tensor([T, P]))

        current_time = 0

        for t in range(1,T):
            if t > current_time:
                current_time += 1

                dSdt = -beta * I[current_time-1].clone()* S[current_time-1].clone()/Npop+ noise_b[t]
                dIdt =  beta * I[current_time-1].clone() * S[current_time-1].clone()/Npop- gamma * I[current_time-1].clone() - noise_b[t] + noise_g[t]
                dRdt =  gamma * I[current_time-1].clone() - noise_g[t]

                S[current_time] = S[current_time-1].clone()+dSdt.clone()
                I[current_time] = I[current_time-1].clone()+dIdt.clone()
                R[current_time] = R[current_time-1].clone()+dRdt.clone()

                idx_constraint = np.where(I[current_time].clone()<0)[0]

            I[current_time][idx_constraint] = 1.
            S[current_time][idx_constraint] = Npop - 1.
 
            lognewWeights = lw.clone() + torch.distributions.poisson.Poisson(I[t].clone()+ 1e-5).log_prob(self.y[t])
            lw = lognewWeights.clone()

            loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)

            wnorm = torch.exp(lw - loglikelihood[t])
            neff = 1. / torch.sum(wnorm * wnorm)

            if neff<P/2:
                idx = rngs.torchMultinomial(P, wnorm)
                S[:] = S[:, idx]
                I[:] = I[:, idx]
                R[:] = R[:, idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)
        
        return loglikelihood[T-1]


