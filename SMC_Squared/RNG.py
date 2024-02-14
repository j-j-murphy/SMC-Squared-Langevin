import numpy as np
import torch

class RNG():

    def __init__(self, seed=0):
        self.seed = seed
        self.nprng = np.random.RandomState(seed)
        #torch randn
        #torch r sample
        #torch multinomial

    def torchRandn(self, size):
        return torch.randn(size)
    
    def torchMultinomial(self, n, w):
        return torch.multinomial(w, n, replacement=True)
    
    def torchNormalrsample(self, tensor, mu=0, sigma=1):
        return torch.distributions.Normal(mu, sigma).rsample(tensor)

    def randomNormal(self, mu=0, sigma=1, size=1):
        if size!=1:
            return self.nprng.normal(loc=mu, scale=sigma, size=size)
        else:
            return self.nprng.normal(loc=mu, scale=sigma)
        
    def randomMVNormal(self, mu, sigma):
        return self.nprng.multivariate_normal(mean=mu, cov=sigma)

    def randomUniform(self, low=0, high=1):
        return self.nprng.uniform(low=low, high=high)

    def randomGamma(self, shape=1, scale=1, size=1):
        return self.nprng.gamma(shape=shape, scale=scale)
        
    def randomMultinomial(self, n, w):
        return self.nprng.multinomial(n=n, pvals=w)

    def randomChoice(self, n1, n2, w):
        return self.nprng.choice(n1, n2, p=w, replace=True)
		
    def randomBinomial(self, x, p):
        return self.nprng.binomial(x, p)
    
    def randomExponential(self, scale, size=1):
        return(self.nprng.exponential(scale, size=1))
        
    def randomUniform(self, a, b, size=1):
        return(self.nprng.uniform(a, b, size=1))
        
    def randomInt(self, a, b, size=1):
        return(self.nprng.uniform(a, b, size=1))
