# 🎲 Monte Carlo Insurance Simulator

A comprehensive actuarial simulation tool using Monte Carlo methods. Covers core **SOA Exam P** concepts with an interactive Streamlit dashboard and full Excel I/O pipeline.

## 🚀 Live Demo
👉 **[Click here to try the app](https://monte-carlo-insurance-simulator-pranavaswaruban-s.streamlit.app/)**

## 📋 Features
- **Distribution Explorer** — Poisson, Binomial, NegBin, Exponential, Gamma, Pareto, Lognormal, Weibull
- **Aggregate Loss Model** — Compound distributions with VaR/TVaR risk measures
- **Ruin Theory** — Cramér-Lundberg surplus process with live simulation
- **CLT Demo** — Visual convergence to Normal
- **Bayes' Theorem** — Risk classification with posterior updating
- **Excel Pipeline** — Upload claims → Simulate → Download reports

## 🛠️ Tech Stack
Python | Streamlit | NumPy | SciPy | Pandas | Plotly | OpenPyXL

## 📦 Quick Start
```bash
git clone https://github.com/YOUR-USERNAME/monte-carlo-insurance-simulator.git
cd monte-carlo-insurance-simulator
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure
```
├── app.py                  # Streamlit dashboard
├── distributions/          # Frequency & Severity models
├── models/                 # Aggregate loss & Ruin theory
├── utils/                  # Excel I/O & sample data
└── data/                   # Sample claims dataset
```

## 📖 Exam P Concepts
Probability Distributions • MGFs • Compound Models • VaR/TVaR • Ruin Theory • CLT • Bayes' Theorem • Memoryless Property • Covariance
