import torch
from torch.autograd import grad

from models.target_pf import Target_PF

class earthquake_PF(Target_PF):  
    def run_particleFilter(self, thetas, rngs):
        T = len(self.y)
        P = self.P
        mu, sigma, beta = thetas # 0.8, 0.15, 17.65
            
        xp = torch.zeros((T, P))
        lw = torch.zeros(P)
        loglikelihood = torch.zeros(T)

        xp[0] = torch.flatten(torch.distributions.Normal(mu, sigma).rsample(torch.tensor([P])))
        lw[:] = -torch.log(torch.ones(1,1)*P)
        
        for t in range(1,T):
            xp[t] = torch.distributions.Normal(mu*xp[t-1].clone(), sigma).rsample()
            lognewWeights = lw.clone() + torch.distributions.poisson.Poisson(beta*torch.exp(xp[t].clone())).log_prob(self.y[t-1])
            lw = lognewWeights.clone()
            loglikelihood[t] = torch.logsumexp(lw.clone(),dim=0)
            
            wnorm = torch.exp(lw-loglikelihood[t]) #normalised weights (on a linear scale)
            neff = 1./torch.sum(wnorm*wnorm)
            
            if(neff<P/2):
                idx = rngs.torchMultinomial(P, wnorm)
                xp[:] = xp[:, idx]
                lw[:] = loglikelihood[t]-torch.log(torch.ones(1,1)*P)
        
        return loglikelihood[T-1]

