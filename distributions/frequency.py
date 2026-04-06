"""
FREQUENCY DISTRIBUTIONS — How many claims happen?
===================================================
Poisson, Binomial, Negative Binomial
"""
import numpy as np
from scipy import stats


class PoissonDist:
    """Poisson(λ): E[X] = Var(X) = λ"""
    def __init__(self, lam):
        self.lam = lam
        self.name = "Poisson"
    def sample(self, n=10000):
        return np.random.poisson(self.lam, size=n)
    def mean(self):
        return self.lam
    def variance(self):
        return self.lam
    def std(self):
        return float(np.sqrt(self.lam))
    def info(self):
        return {"Distribution": self.name, "λ": self.lam, "Mean": self.lam,
                "Variance": self.lam, "Std Dev": self.std()}


class BinomialDist:
    """Binomial(n, p): E[X] = np, Var(X) = np(1-p)"""
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.name = "Binomial"
    def sample(self, size=10000):
        return np.random.binomial(self.n, self.p, size=size)
    def mean(self):
        return self.n * self.p
    def variance(self):
        return self.n * self.p * (1 - self.p)
    def std(self):
        return float(np.sqrt(self.variance()))
    def info(self):
        return {"Distribution": self.name, "n": self.n, "p": self.p,
                "Mean": self.mean(), "Variance": self.variance(), "Std Dev": self.std()}


class NegBinomialDist:
    """NegBinomial(r, p): E[X] = r(1-p)/p, Var(X) = r(1-p)/p²"""
    def __init__(self, r, p):
        self.r = r
        self.p = p
        self.name = "Negative Binomial"
    def sample(self, size=10000):
        return np.random.negative_binomial(self.r, self.p, size=size)
    def mean(self):
        return self.r * (1 - self.p) / self.p
    def variance(self):
        return self.r * (1 - self.p) / (self.p ** 2)
    def std(self):
        return float(np.sqrt(self.variance()))
    def info(self):
        return {"Distribution": self.name, "r": self.r, "p": self.p,
                "Mean": self.mean(), "Variance": self.variance(), "Std Dev": self.std()}
