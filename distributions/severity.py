"""
SEVERITY DISTRIBUTIONS — How big is each claim?
=================================================
Exponential, Gamma, Pareto, Lognormal, Weibull
"""
import numpy as np


class ExponentialDist:
    """Exponential(λ): Mean = 1/λ, Memoryless"""
    def __init__(self, lam):
        self.lam = lam
        self.name = "Exponential"
    def sample(self, n=10000):
        return np.random.exponential(1.0 / self.lam, size=n)
    def mean(self):
        return 1.0 / self.lam
    def variance(self):
        return 1.0 / (self.lam ** 2)
    def std(self):
        return 1.0 / self.lam
    def info(self):
        return {"Distribution": self.name, "λ": self.lam, "Mean": self.mean(),
                "Variance": self.variance()}


class GammaDist:
    """Gamma(α, β): Mean = α/β"""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.name = "Gamma"
    def sample(self, n=10000):
        return np.random.gamma(self.alpha, 1.0 / self.beta, size=n)
    def mean(self):
        return self.alpha / self.beta
    def variance(self):
        return self.alpha / (self.beta ** 2)
    def std(self):
        return float(np.sqrt(self.variance()))
    def info(self):
        return {"Distribution": self.name, "α": self.alpha, "β": self.beta,
                "Mean": self.mean(), "Variance": self.variance()}


class ParetoDist:
    """Pareto(α, θ): Mean = θ/(α-1), Heavy tail"""
    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta
        self.name = "Pareto"
    def sample(self, n=10000):
        u = np.random.uniform(0, 1, n)
        return self.theta * (u ** (-1.0 / self.alpha) - 1)
    def mean(self):
        return self.theta / (self.alpha - 1) if self.alpha > 1 else float('inf')
    def variance(self):
        if self.alpha <= 2:
            return float('inf')
        return (self.alpha * self.theta**2) / ((self.alpha - 1)**2 * (self.alpha - 2))
    def std(self):
        v = self.variance()
        return float(np.sqrt(v)) if v != float('inf') else float('inf')
    def info(self):
        return {"Distribution": self.name, "α": self.alpha, "θ": self.theta,
                "Mean": self.mean(), "Tail": "Heavy"}


class LognormalDist:
    """Lognormal(μ, σ): Mean = exp(μ + σ²/2)"""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.name = "Lognormal"
    def sample(self, n=10000):
        return np.random.lognormal(self.mu, self.sigma, size=n)
    def mean(self):
        return float(np.exp(self.mu + self.sigma**2 / 2))
    def variance(self):
        return float((np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2))
    def std(self):
        return float(np.sqrt(self.variance()))
    def info(self):
        return {"Distribution": self.name, "μ": self.mu, "σ": self.sigma,
                "Mean": self.mean(), "Median": float(np.exp(self.mu))}


class WeibullDist:
    """Weibull(k, λ): k<1 decreasing hazard, k=1 Exponential, k>1 increasing"""
    def __init__(self, k, lam):
        self.k = k
        self.lam = lam
        self.name = "Weibull"
    def sample(self, n=10000):
        return self.lam * np.random.weibull(self.k, size=n)
    def mean(self):
        from scipy.special import gamma
        return float(self.lam * gamma(1 + 1.0 / self.k))
    def variance(self):
        from scipy.special import gamma
        return float(self.lam**2 * (gamma(1 + 2.0/self.k) - gamma(1 + 1.0/self.k)**2))
    def std(self):
        return float(np.sqrt(self.variance()))
    def info(self):
        return {"Distribution": self.name, "k": self.k, "λ": self.lam,
                "Mean": self.mean()}
