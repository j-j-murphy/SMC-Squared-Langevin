import numpy as np
import sys
sys.path.append('..')  # noqa

import torch
from torch.autograd import grad

from models.target_pf import Target_PF


def generateData(theta, noObservations, initialState):
    mu, phi, sigmav = theta
    state = np.zeros(noObservations + 1)
    observation = np.zeros(noObservations)
    state[0] = initialState
    for t in range(1, noObservations):
        state[t] = torch.distributions.Normal(mu * state[t-1], phi).sample() # mu * state[t - 1] + phi * np.random.randn()
        observation[t] = torch.distributions.Normal(state[t], sigmav).sample()

    return(state, observation)

class lgssm_PF(Target_PF):   
    def run_particleFilter(self, thetas, rngs):
        T = len(self.y)-1
        P = self.P
        mu, phi, sigma = thetas

        xp = torch.zeros((T, P))
        lw = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        xp[0] = torch.full((1,P),  0)[0]+rngs.torchRandn(P)
        lw[:] = -torch.log(torch.ones(1,1)*P)

        for t in range(1,T):
            part1 = (phi**(-2) + sigma**(-2))**(-1)
            part2 = sigma**(-2) * self.y[t]
            part2 = part2 + phi**(-2) * mu * xp[t-1].clone()

            xp[t] = part1 * part2.clone() + torch.sqrt(part1.clone()) * rngs.torchNormalrsample(torch.tensor([P])) #noise[t]

            yhatMean = mu * xp[t].clone()
            yhatVariance = torch.sqrt(phi**2 + sigma**2)
            lognewWeights = lw.clone() + torch.distributions.Normal(yhatMean.clone(), yhatVariance).log_prob(torch.tensor([self.y[t+1]]))

            lw = lognewWeights.clone()
            loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)

            wnorm = torch.exp(lw-loglikelihood[t]) #normalised weights (on a linear scale)
            neff = 1./torch.sum(wnorm*wnorm)

            if(neff<P/2):
                idx = rngs.torchMultinomial(P, wnorm)
                xp[:] = xp[:, idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)

        return loglikelihood[T-1]

