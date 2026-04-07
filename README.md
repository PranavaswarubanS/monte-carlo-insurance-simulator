<div align="center">

# 🎲 Monte Carlo Insurance Simulator

### *An interactive actuarial dashboard for loss distribution analysis*

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Open_App-f97316?style=for-the-badge&logoColor=white)](https://monte-carlo-insurance-simulator-pranavaswaruban-s.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge)](LICENSE)

**Built with Python • Streamlit • NumPy • SciPy • Plotly • OpenPyXL**

[**🚀 Try the Live Demo →**](https://monte-carlo-insurance-simulator-pranavaswaruban-s.streamlit.app/)

</div>

---

## 🎯 What Is This?

A comprehensive **Monte Carlo simulation tool** that helps insurance companies model loss distributions and answer the most critical question in their business:

> *"How much money do we need to keep in reserve so we don't go bankrupt?"*

Insurance claims are doubly random — both the **number** of claims and the **size** of each claim are uncertain. This project simulates **10,000 possible years** of insurance operations to estimate total losses, risk metrics, and ruin probabilities.

Built as a portfolio project covering all core **SOA Exam P** concepts.

---

## ✨ Features

| Module | What It Does |
|:------:|:-------------|
| 📊 **Distribution Explorer** | Interactive Poisson, Binomial, Negative Binomial, Exponential, Gamma, Pareto, Lognormal, and Weibull with live parameter sliders |
| 🏗️ **Aggregate Loss Model** | Compound distribution `S = X₁ + X₂ + ... + X_N` with VaR and TVaR risk measures |
| ⚠️ **Ruin Theory** | Cramér-Lundberg surplus process with live path simulation and exact formula validation |
| 📐 **CLT Demonstration** | Visual convergence to Normal distribution as sample size grows |
| 🔮 **Bayes' Theorem** | Risk classification with prior/posterior updating — simulated vs. exact |
| 📗 **Excel Pipeline** | Upload claim data → Run simulation → Download formatted multi-sheet report |

---

## 🚀 Live Demo

**👉 [https://monte-carlo-insurance-simulator-pranavaswaruban-s.streamlit.app/](https://monte-carlo-insurance-simulator-pranavaswaruban-s.streamlit.app/)**

No installation required — click the link, drag the sliders, run simulations, download Excel reports.

---

## 📊 Core Math

This project implements the standard actuarial framework:

```
Aggregate Loss:    S = X₁ + X₂ + ... + X_N

Expected Value:    E[S] = E[N] · E[X]

Variance:          Var(S) = E[N]·Var(X) + Var(N)·(E[X])²

Surplus Process:   U(t) = u + ct − S(t)

Ruin Probability:  ψ(u) = P(U(t) < 0 for some t > 0)
```

### Risk Measures
- **VaR (Value at Risk):** 95th percentile of losses → "worst case in 95% of scenarios"
- **TVaR (Tail VaR):** Average of worst 5% → "when things go bad, how bad?"

---

## 🛠️ Tech Stack

| Tool | Role |
|:----:|:-----|
| **Python 3.9+** | Core language |
| **Streamlit** | Interactive web dashboard |
| **NumPy** | Random number generation & fast math |
| **SciPy** | Probability distributions & statistical tests |
| **Pandas** | Data handling & Excel I/O |
| **Plotly** | Interactive charts |
| **OpenPyXL** | Excel file creation |

---

## 📦 Run Locally

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/PranavaswarubanS/monte-carlo-insurance-simulator.git
cd monte-carlo-insurance-simulator

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
monte-carlo-insurance-simulator/
│
├── app.py                       # Main Streamlit dashboard
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── distributions/
│   ├── frequency.py             # Poisson, Binomial, NegBin
│   └── severity.py              # Exponential, Gamma, Pareto, Lognormal, Weibull
│
├── models/
│   ├── aggregate.py             # Monte Carlo aggregate loss engine
│   └── ruin.py                  # Cramér-Lundberg ruin simulation
│
├── utils/
│   ├── excel_io.py              # Excel read/write functions
│   └── sample_data.py           # Synthetic claims generator
│
└── data/
    └── sample_claims.xlsx       # 5,000 sample insurance claims
```

---

## 📖 SOA Exam P Concepts Covered

<table>
<tr>
<td>

**Probability Distributions**
- Poisson, Binomial
- Negative Binomial
- Exponential, Gamma
- Pareto, Lognormal, Weibull

</td>
<td>

**Risk Theory**
- Aggregate Loss Models
- Compound Distributions
- VaR and TVaR
- Ruin Theory
- Safety Loading

</td>
<td>

**Foundational Concepts**
- Moment Generating Functions
- Central Limit Theorem
- Bayes' Theorem
- Memoryless Property
- Joint Distributions

</td>
</tr>
</table>

---

## 💡 How to Use the Dashboard

1. **📊 Distributions Tab** — Pick a frequency or severity distribution. Drag sliders to see how parameters affect the shape and statistics.

2. **🏗️ Aggregate Loss Tab** — Set frequency and severity parameters, run a Monte Carlo simulation, and see the full loss distribution with VaR/TVaR markers.

3. **⚠️ Ruin Theory Tab** — Adjust initial surplus, premium rate, and claim parameters. Watch surplus paths in real time and see how often the company goes bankrupt.

4. **📐 CLT Tab** — Slide the sample size from `n=1` to `n=100` and watch a heavily skewed Exponential distribution become a perfect bell curve.

5. **🔮 Bayes Tab** — See how prior risk classifications update after observing claim data.

6. **📗 Excel Tab** — Upload your own claims data or generate a sample dataset, then download a professional multi-sheet Excel report.

---

## 🎓 Learning Objectives

This project demonstrates:
- ✅ **Statistical modeling** with multiple probability distributions
- ✅ **Monte Carlo simulation** techniques for problems without closed-form solutions
- ✅ **Risk quantification** using VaR and TVaR
- ✅ **Data pipeline design** (Excel → Python → Streamlit → Excel)
- ✅ **Interactive data visualization** with Plotly
- ✅ **Web app deployment** using Streamlit Cloud
- ✅ **Software engineering** with modular Python packages

---

## 👤 Author

**Pranavaswaruban S**
Actuarial Science Student | SOA Exam P Candidate

🐙 GitHub: [@PranavaswarubanS](https://github.com/PranavaswarubanS)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

### ⭐ If you found this useful, please give it a star!

**Built with ❤️ and Python**

</div>
