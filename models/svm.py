import numpy as np
import sys
sys.path.append('..')  # noqa

import torch
from torch.autograd import grad

from models.target_pf import Target_PF

def generateData(theta, noObservations, initialState):
    mu, sigma_x, sigma_y = theta
    state = torch.zeros(noObservations + 1)
    observation = torch.zeros(noObservations)
    state[0] = initialState  # Initial log-volatility
    for t in range(1, noObservations):
        state[t] = torch.distributions.Normal(mu * state[t - 1], sigma_x).sample()
        observation[t] = torch.distributions.Normal(0, sigma_y**2 * torch.exp(state[t])).sample()

    return state, observation

class svm_PF(Target_PF):   
    def run_particleFilter(self, thetas, rngs):
        T = len(self.y)
        P = self.P
        mu, sigma_x, sigma_y = thetas

        xp = torch.zeros((T, P))  # Particles for log-volatility
        lw = torch.zeros(P)       # Log-weights
        loglikelihood = torch.zeros(T)

        xp[0] = torch.full((1, P), 0.0)[0] + rngs.torchRandn(P)
        lw[:] = -torch.log(torch.ones(1, 1) * P)
        noise = rngs.torchNormalrsample(torch.tensor([T, P]))

        for t in range(1, T):
            xp[t] = mu * xp[t - 1].clone() + sigma_x * noise[t]
            yhatVariance = sigma_y**2 * torch.exp(xp[t].clone())  # exp(h_t / 2) is the standard deviation of y_t
            lognewWeights = lw.clone() + torch.distributions.Normal(0, yhatVariance).log_prob(self.y[t])

            lw = lognewWeights.clone()
            loglikelihood[t] = torch.logsumexp(lw.clone(), dim=0)

            wnorm = torch.exp(lw - loglikelihood[t])
            neff = 1. / torch.sum(wnorm * wnorm)

            if neff < P / 2:
                idx = rngs.torchMultinomial(P, wnorm)
                xp[:] = xp[:, idx]
                lw[:] = loglikelihood[t] - torch.log(torch.ones(1, 1) * P)

        return loglikelihood[T - 1]

