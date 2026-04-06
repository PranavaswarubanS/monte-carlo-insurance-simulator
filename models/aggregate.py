"""
AGGREGATE LOSS MODEL — S = X1 + X2 + ... + XN
================================================
E[S] = E[N] * E[X]
Var(S) = E[N]*Var(X) + Var(N)*(E[X])^2
"""
import numpy as np


class AggregateLossSimulator:
    def __init__(self, freq_dist, sev_dist):
        self.freq = freq_dist
        self.sev = sev_dist

    def simulate(self, n_simulations=10000):
        counts = self.freq.sample(n_simulations)
        losses = np.zeros(n_simulations)
        for i in range(n_simulations):
            nc = int(counts[i])
            if nc > 0:
                losses[i] = float(np.sum(self.sev.sample(nc)))

        risk = {}
        for a in [0.90, 0.95, 0.975, 0.99]:
            v = float(np.percentile(losses, a * 100))
            tail = losses[losses >= v]
            risk[f"VaR_{a}"] = v
            risk[f"TVaR_{a}"] = float(np.mean(tail)) if len(tail) > 0 else v

        e_n = self.freq.mean()
        var_n = self.freq.variance()
        e_x = self.sev.mean()
        var_x = self.sev.variance()
        e_s = e_n * e_x if e_x != float('inf') else float('inf')
        var_s = (e_n * var_x + var_n * e_x**2) if (var_x != float('inf') and e_x != float('inf')) else float('inf')

        return {
            "losses": losses,
            "counts": counts,
            "risk": risk,
            "mean": float(np.mean(losses)),
            "std": float(np.std(losses)),
            "median": float(np.median(losses)),
            "skew": float(np.mean(((losses - np.mean(losses)) / max(np.std(losses), 1e-10))**3)),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "mean_claims": float(np.mean(counts)),
            "analytical": {"E[N]": e_n, "Var(N)": var_n, "E[X]": e_x, "Var(X)": var_x, "E[S]": e_s, "Var(S)": var_s},
            "n_sim": n_simulations,
        }
