import numpy as np
import sys
sys.path.append('..')  # noqa
from mpi4py import MPI

from scipy.stats import multivariate_normal as Normal_PDF
from smc_components.SMC_TEMPLATES import  Q_Base

class Q(Q_Base):
    """ Define general proposal """
    def __init__(self, step_size, prop, ss_2nd_1st=0.09):
        self.step_size = step_size
        self.prop = prop
        self.ss_2nd_1st = ss_2nd_1st

    def pdf(self, x, x_cond):
        return (2 * np.pi)**-0.5 * np.exp(-0.5 * (x - x_cond).T @ (x - x_cond))

    def logpdf(self, x, x_cond, v, grads, hess):
        if self.prop == 'rw':
            logpdf = Normal_PDF(mean=x_cond, cov=self.step_size**2 * np.eye(len(x_cond))).logpdf(x)

        elif self.prop == 'first_order':
            logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=np.eye(len(x_cond))).logpdf(v)
            # logpdf = Normal_PDF(mean=x_cond+self.step_size**2/2*grads, cov=self.step_size**2 * np.eye(len(x_cond))).logpdf(x)

        elif self.prop == 'second_order':
            neg_hess = -hess
            if self.isPSD(neg_hess):
                try:
                    inv_neg_hess = np.linalg.pinv(neg_hess)
                    jac = self.step_size**len(inv_neg_hess)*np.prod(np.diag(inv_neg_hess))
                    logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=inv_neg_hess, allow_singular=True).logpdf(v) * jac
                except np.linalg.LinAlgError:
                    print(" falling back to first-order.")
                    step_size = self.ss_2nd_1st
                    logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=np.eye(len(x))).logpdf(v)
                # logpdf = Normal_PDF(mean=x_cond+self.step_size**2/2*np.dot(grads, inv_neg_hess), cov=self.step_size**2 * inv_neg_hess).logpdf(x)
            else:
                step_size = self.ss_2nd_1st # Reverting to first-order step-size
                logpdf = Normal_PDF(mean=np.zeros(len(x)), cov=np.eye(len(x_cond))).logpdf(v)
                # logpdf = Normal_PDF(mean=x_cond+step_size**2/2*grads, cov=step_size**2 * np.eye(len(x_cond))).logpdf(x)

        return logpdf
    
    def rvs(self, x_cond, rngs, grads, hess):
        if self.prop == 'rw':
            v = rngs.randomMVNormal(np.zeros(len(x_cond)), np.eye(len(x_cond)))
            v_half = v
            x_new = x_cond + self.step_size*v
            # v, v_new = None, None
            # x_new = rngs.randomMVNormal(x_cond, self.step_size**2 * np.eye(len(x_cond)))

        elif self.prop == 'first_order':
            v = rngs.randomMVNormal(np.zeros(len(x_cond)), np.eye(len(x_cond)))
            v_half = 0.5 * self.step_size*grads + v
            x_new = x_cond + self.step_size*v_half
            # v_new = 0.5 * self.step_size*grads + v
            # v, v_new = None, None
            # x_new = rngs.randomMVNormal(x_cond+self.step_size**2/2*grads, self.step_size**2 * np.eye(len(x_cond)))

        elif self.prop == 'second_order':
            neg_hess = -hess
            if self.isPSD(neg_hess):
                inv_neg_hess = np.linalg.pinv(neg_hess)
                v = rngs.randomMVNormal(np.zeros(len(x_cond)), neg_hess)
                v_half = 0.5*self.step_size*grads + v
                x_new = x_cond + self.step_size*np.dot(inv_neg_hess, v_half)
                # x_new = rngs.randomMVNormal(x_cond+self.step_size**2/2*np.dot(grads, inv_neg_hess), self.step_size**2 * inv_neg_hess)
            else:
                print("NOT PSD")
                step_size = self.ss_2nd_1st # Reverting to first-order step-size
                v = rngs.randomMVNormal(np.zeros(len(x_cond)), np.eye(len(x_cond)))
                v_half = 0.5 * step_size*grads + v
                x_new = x_cond + step_size*v_half
                # v, v_new = None, None
                # x_new = rngs.randomMVNormal(x_cond+step_size**2/2*grads, step_size**2 * np.eye(len(x_cond)))

        return x_new, v, v_half
    
    def isPSD(self, x):
        try:
            return np.all(np.linalg.eigvals(x) > 0)
        except:
            return False
