"""
RUIN THEORY — Cramér-Lundberg Surplus Process
===============================================
U(t) = u + ct - S(t)
Ruin when U(t) < 0
"""
import numpy as np


class RuinSimulator:
    def __init__(self, initial_surplus, premium_rate, claim_freq, sev_dist):
        self.u = initial_surplus
        self.c = premium_rate
        self.lam = claim_freq
        self.sev = sev_dist

    def simulate(self, n_paths=500, horizon=30, dt=0.05):
        n_steps = int(horizon / dt)
        ruin_count = 0
        paths = []
        step_skip = max(1, n_steps // 400)

        for i in range(n_paths):
            s = self.u
            path = [s] if i < 20 else None
            ruined = False

            for j in range(1, n_steps):
                s += self.c * dt
                nc = np.random.poisson(self.lam * dt)
                if nc > 0:
                    claims = self.sev.sample(nc)
                    s -= float(np.sum(claims))

                if path is not None and j % step_skip == 0:
                    path.append(s)

                if s < 0 and not ruined:
                    ruined = True
                    ruin_count += 1

            if path is not None:
                paths.append(path)

        time_pts = [round(j * dt * step_skip, 2) for j in range(len(paths[0]))] if paths else []

        mean_claim = self.sev.mean()
        exp_claims = self.lam * mean_claim
        safety = (self.c - exp_claims) / exp_claims if exp_claims > 0 else 0

        # Exact formula for Exponential severity
        exact = None
        try:
            if self.c > exp_claims:
                rho = exp_claims / self.c
                exact = float(rho * np.exp(-(1 - rho) * self.u / mean_claim))
            else:
                exact = 1.0
        except:
            pass

        return {
            "ruin_prob": ruin_count / n_paths,
            "n_ruined": ruin_count,
            "n_paths": n_paths,
            "safety_loading": safety,
            "surplus_0": self.u,
            "premium": self.c,
            "exp_claims": exp_claims,
            "net_profit_ok": self.c > exp_claims,
            "exact_ruin": exact,
            "time": time_pts,
            "paths": paths,
        }
