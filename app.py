"""
🎲 MONTE CARLO INSURANCE SIMULATOR
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

from distributions import PoissonDist, BinomialDist, NegBinomialDist
from distributions import ExponentialDist, GammaDist, ParetoDist, LognormalDist, WeibullDist
from models import AggregateLossSimulator, RuinSimulator
from utils import read_claims_excel, export_report, generate_sample_dataset

# ── Config ──
st.set_page_config(page_title="Monte Carlo Insurance Simulator", page_icon="🎲", layout="wide")
st.title("🎲 Monte Carlo Insurance Simulator")
st.caption("Loss Distribution Dashboard — SOA Exam P Concepts")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Distributions", "🏗️ Aggregate Loss", "⚠️ Ruin Theory",
    "📐 CLT Demo", "🔮 Bayes' Theorem", "📗 Excel I/O"])

# ══════════════════════════════════════
# TAB 1: DISTRIBUTIONS
# ══════════════════════════════════════
with tab1:
    cf, cs = st.columns(2)

    with cf:
        st.subheader("📊 Frequency — How Many Claims?")
        ft = st.selectbox("Distribution", ["Poisson", "Binomial", "Negative Binomial"], key="ft")
        if ft == "Poisson":
            lam = st.slider("λ (avg claims/year)", 1.0, 20.0, 5.0, 0.5, key="fl")
            fd = PoissonDist(lam)
            st.latex(r"P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} \quad E[X]=Var(X)=\lambda")
        elif ft == "Binomial":
            n = st.slider("n (policies)", 10, 100, 50, key="fn")
            p = st.slider("p (claim prob)", 0.01, 0.5, 0.1, 0.01, key="fp")
            fd = BinomialDist(n, p)
            st.latex(r"P(X=k) = \binom{n}{k}p^k(1-p)^{n-k} \quad E[X]=np")
        else:
            r = st.slider("r (successes)", 1, 10, 3, key="fr")
            p = st.slider("p (prob)", 0.1, 0.9, 0.4, 0.05, key="fp2")
            fd = NegBinomialDist(r, p)
            st.latex(r"E[X] = \frac{r(1-p)}{p} \quad Var(X) = \frac{r(1-p)}{p^2}")

        fs = fd.sample(10000)
        fi = fd.info()
        c1, c2, c3 = st.columns(3)
        c1.metric("E[N]", f"{fi['Mean']:.2f}")
        c2.metric("Var(N)", f"{fi['Variance']:.2f}")
        c3.metric("σ(N)", f"{fi['Std Dev']:.2f}")
        fig = px.histogram(x=fs, nbins=int(max(fs)-min(fs)+1),
                          title=f"{ft} (10K samples)", color_discrete_sequence=["#3b82f6"])
        fig.update_layout(bargap=0.1, height=340, xaxis_title="Claims", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with cs:
        st.subheader("💰 Severity — Claim Size")
        svt = st.selectbox("Distribution", ["Lognormal","Exponential","Gamma","Pareto","Weibull"], key="svt")
        if svt == "Exponential":
            sr = st.slider("λ (rate)", 0.0002, 0.01, 0.001, 0.0002, format="%.4f", key="sl")
            sd = ExponentialDist(sr)
            st.latex(r"f(x) = \lambda e^{-\lambda x} \quad E[X]=\frac{1}{\lambda}")
        elif svt == "Gamma":
            sa = st.slider("α (shape)", 0.5, 10.0, 2.0, 0.5, key="sa")
            sb = st.slider("β (rate)", 0.0005, 0.01, 0.002, 0.0005, format="%.4f", key="sb")
            sd = GammaDist(sa, sb)
            st.latex(r"E[X] = \frac{\alpha}{\beta} \quad Var(X) = \frac{\alpha}{\beta^2}")
        elif svt == "Pareto":
            spa = st.slider("α (shape)", 1.5, 10.0, 3.0, 0.5, key="spa")
            spt = st.slider("θ (scale)", 500, 20000, 5000, 500, key="spt")
            sd = ParetoDist(spa, spt)
            st.latex(r"f(x) = \frac{\alpha\theta^\alpha}{(\theta+x)^{\alpha+1}} \quad \text{Heavy Tail}")
        elif svt == "Lognormal":
            smu = st.slider("μ (log-mean)", 4.0, 10.0, 7.0, 0.5, key="smu")
            ssg = st.slider("σ (log-std)", 0.3, 2.5, 1.2, 0.1, key="ssg")
            sd = LognormalDist(smu, ssg)
            st.latex(r"X = e^Y,\ Y \sim N(\mu,\sigma^2) \quad E[X]=e^{\mu+\sigma^2/2}")
        else:
            sk = st.slider("k (shape)", 0.5, 5.0, 1.5, 0.1, key="sk")
            swl = st.slider("λ (scale)", 100, 5000, 1000, 100, key="swl")
            sd = WeibullDist(sk, swl)
            st.latex(r"k<1: \downarrow hazard \quad k=1: Exp \quad k>1: \uparrow hazard")

        ss = sd.sample(10000)
        mv = sd.mean()
        p95 = float(np.percentile(ss, 95))
        c1, c2, c3 = st.columns(3)
        c1.metric("E[X]", f"${mv:,.0f}" if mv < 1e9 else "∞")
        c2.metric("VaR 95%", f"${p95:,.0f}")
        c3.metric("Median", f"${np.median(ss):,.0f}")
        fig = px.histogram(x=ss, nbins=80, title=f"{svt} Severity (10K samples)",
                          color_discrete_sequence=["#8b5cf6"])
        fig.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
        fig.update_layout(height=340, xaxis_title="Claim ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════
# TAB 2: AGGREGATE LOSS
# ══════════════════════════════════════
with tab2:
    st.subheader("🏗️ Aggregate Loss: S = X₁ + X₂ + ... + X_N")
    st.latex(r"E[S] = E[N] \cdot E[X] \qquad Var(S) = E[N]Var(X) + Var(N)(E[X])^2")

    cp, cr = st.columns([1, 2])
    with cp:
        al = st.slider("λ (Poisson freq)", 1.0, 20.0, 5.0, 0.5, key="al")
        am = st.slider("μ (Lognormal)", 4.0, 10.0, 7.0, 0.5, key="am")
        asig = st.slider("σ (Lognormal)", 0.3, 2.5, 1.2, 0.1, key="as")
        ansim = st.slider("Simulations", 1000, 20000, 10000, 1000, key="an")
        arun = st.button("▶ Run Simulation", type="primary", key="arun")

    with cr:
        if arun or "ar" not in st.session_state:
            sim = AggregateLossSimulator(PoissonDist(al), LognormalDist(am, asig))
            st.session_state["ar"] = sim.simulate(ansim)
            st.session_state["ap"] = {"Freq": f"Poisson(λ={al})", "Sev": f"Lognormal(μ={am},σ={asig})", "N": ansim}

        ar = st.session_state.get("ar")
        if ar:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("E[S]", f"${ar['mean']:,.0f}")
            m2.metric("σ(S)", f"${ar['std']:,.0f}")
            m3.metric("VaR 95%", f"${ar['risk']['VaR_0.95']:,.0f}")
            m4.metric("TVaR 95%", f"${ar['risk']['TVaR_0.95']:,.0f}")

            fig = px.histogram(x=ar["losses"], nbins=80, title="Aggregate Loss Distribution",
                              color_discrete_sequence=["#10b981"])
            fig.add_vline(x=ar["risk"]["VaR_0.95"], line_dash="dash", line_color="orange", annotation_text="VaR 95%")
            fig.add_vline(x=ar["risk"]["TVaR_0.95"], line_dash="dash", line_color="red", annotation_text="TVaR 95%")
            fig.update_layout(height=400, xaxis_title="Total Loss ($)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📐 Analytical vs Simulated"):
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown("**Formula Results**")
                    for k, v in ar["analytical"].items():
                        st.write(f"**{k}**: {v:,.2f}" if isinstance(v, (int, float)) and v < 1e15 else f"**{k}**: {v}")
                with ac2:
                    st.markdown("**Simulation Results**")
                    st.write(f"**Mean**: ${ar['mean']:,.2f}")
                    st.write(f"**Std Dev**: ${ar['std']:,.2f}")
                    st.write(f"**Skewness**: {ar['skew']:.4f}")


# ══════════════════════════════════════
# TAB 3: RUIN THEORY — AUTO-REACTIVE
# ══════════════════════════════════════
with tab3:
    st.subheader("⚠️ Ruin Theory — Cramér-Lundberg Surplus Process")
    st.latex(r"U(t) = u + ct - S(t) \qquad \psi(u) = P(\text{Ruin})")

    rp, rr = st.columns([1, 2])
    with rp:
        ru = st.slider("u — Initial Surplus ($)", 5000, 200000, 15000, 5000, key="ru")
        rc = st.slider("c — Premium Rate ($/yr)", 5000, 50000, 11000, 1000, key="rc")
        rl = st.slider("λ — Claims per Year", 1.0, 20.0, 8.0, 0.5, key="rl")
        rrate = st.slider("Severity λ (Exponential)", 0.0002, 0.005, 0.0007, 0.0001, format="%.4f", key="rr")
        rpaths = st.slider("Number of Paths", 100, 1000, 500, 100, key="rpn")

        st.markdown("---")
        mean_claim = 1.0 / rrate
        exp_claims = rl * mean_claim
        st.markdown(f"**Mean claim size:** ${mean_claim:,.0f}")
        st.markdown(f"**Expected claims/yr:** ${exp_claims:,.0f}")
        st.markdown(f"**Premium/yr:** ${rc:,.0f}")
        if rc > exp_claims:
            st.success(f"✅ Profit margin: ${rc - exp_claims:,.0f}/yr")
        else:
            st.error(f"❌ LOSING ${exp_claims - rc:,.0f}/yr — ruin guaranteed!")

    with rr:
        # Auto-run on every slider change
        sev_r = ExponentialDist(rrate)
        rsim = RuinSimulator(ru, rc, rl, sev_r)
        rres = rsim.simulate(n_paths=rpaths, horizon=30)

        rm1, rm2, rm3 = st.columns(3)
        ruin_pct = rres["ruin_prob"]
        rm1.metric("ψ(u) Simulated", f"{ruin_pct:.4f}", delta=f"{rres['n_ruined']}/{rres['n_paths']} ruined")
        if rres["exact_ruin"] is not None:
            rm2.metric("ψ(u) Exact (Exp)", f"{rres['exact_ruin']:.4f}")
        rm3.metric("Safety Loading θ", f"{rres['safety_loading']:.1%}")

        fig = go.Figure()
        clrs = ["#3b82f6","#8b5cf6","#ec4899","#10b981","#f59e0b","#ef4444",
                "#06b6d4","#84cc16","#f97316","#6366f1","#14b8a6","#e879f9",
                "#22d3ee","#a3e635","#fb7185","#38bdf8","#c084fc","#4ade80","#fbbf24","#f43f5e"]
        for i, path in enumerate(rres["paths"]):
            t = np.linspace(0, 30, len(path))
            fig.add_trace(go.Scatter(x=t, y=path, mode="lines",
                                    line=dict(width=1.3, color=clrs[i % len(clrs)]),
                                    showlegend=False, opacity=0.6))
        fig.add_hline(y=0, line_dash="solid", line_color="#ef4444", line_width=3,
                     annotation_text="RUIN BARRIER", annotation_font_color="#ef4444")
        fig.update_layout(title="Surplus Paths U(t) Over 30 Years",
                         xaxis_title="Time (years)", yaxis_title="Surplus ($)", height=440)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════
# TAB 4: CLT
# ══════════════════════════════════════
with tab4:
    st.subheader("📐 Central Limit Theorem")
    st.latex(r"\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)")
    st.markdown("**Source:** Exponential(λ=0.5) — right-skewed. Watch it become Normal as n grows.")

    cn = st.select_slider("Sample size n", [1,2,5,10,30,50,100], value=1, key="cn")
    means = np.array([np.mean(np.random.exponential(2, cn)) for _ in range(5000)])

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Mean of X̄", f"{np.mean(means):.3f}", delta=f"Theory: 2.000")
    cm2.metric("Std of X̄", f"{np.std(means):.3f}", delta=f"σ/√n = {2/np.sqrt(cn):.3f}")
    cm3.metric("n", cn)

    fig = px.histogram(x=means, nbins=60, title=f"Sample Mean Distribution (n={cn})",
                      color_discrete_sequence=["#14b8a6"])
    fig.update_layout(height=380, xaxis_title="Sample Mean", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    if cn >= 30: st.success("✅ n ≥ 30 — approximately Normal! CLT in action.")
    elif cn >= 10: st.info("🔄 Getting closer to Normal...")
    else: st.warning("⚠️ Small n — original Exponential shape still visible.")


# ══════════════════════════════════════
# TAB 5: BAYES
# ══════════════════════════════════════
with tab5:
    st.subheader("🔮 Bayes' Theorem — Risk Classification")
    st.latex(r"P(H|3+) = \frac{P(3+|H) \cdot P(H)}{P(3+)}")
    st.markdown("**High Risk** (30%, λ=5) vs **Low Risk** (70%, λ=1). Given 3+ claims → how likely High Risk?")

    from scipy.stats import poisson as pois
    pH = 0.3
    p3H = 1 - pois.cdf(2, 5)
    p3L = 1 - pois.cdf(2, 1)
    p3 = p3H * pH + p3L * 0.7
    post = p3H * pH / p3

    # Simulation
    is_h = np.random.binomial(1, pH, 50000).astype(bool)
    cl = np.where(is_h, np.random.poisson(5, 50000), np.random.poisson(1, 50000))
    m3 = cl >= 3
    ps = np.mean(is_h[m3]) if m3.sum() > 0 else 0

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("P(H|3+) Exact", f"{post:.1%}")
    b2.metric("P(H|3+) Simulated", f"{ps:.1%}")
    b3.metric("P(3+|High)", f"{p3H:.1%}")
    b4.metric("P(3+|Low)", f"{p3L:.1%}")

    xv = list(range(15))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xv, y=[pois.pmf(k,5) for k in xv], name="High Risk (λ=5)", marker_color="#ef4444", opacity=0.7))
    fig.add_trace(go.Bar(x=xv, y=[pois.pmf(k,1) for k in xv], name="Low Risk (λ=1)", marker_color="#22c55e", opacity=0.7))
    fig.update_layout(barmode="group", height=380, xaxis_title="Claims", yaxis_title="Probability",
                     title="Claim Distribution by Risk Class")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"**Bayes:** {p3H:.3f} × 0.3 / {p3:.3f} = **{post:.1%}** — filing 3+ claims makes High Risk ~{post:.0%} likely!")


# ══════════════════════════════════════
# TAB 6: EXCEL
# ══════════════════════════════════════
with tab6:
    st.subheader("📗 Excel — Upload Data & Download Reports")
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("### 📥 Upload Claims")
        up = st.file_uploader("Upload .xlsx", type=["xlsx"], key="up")
        if up:
            try:
                df, am, col = read_claims_excel(up)
                st.success(f"✅ {len(am)} claims from '{col}'")
                st.dataframe(df.head(10), use_container_width=True)
                q1,q2,q3 = st.columns(3)
                q1.metric("Mean", f"${np.mean(am):,.0f}")
                q2.metric("Median", f"${np.median(am):,.0f}")
                q3.metric("Max", f"${np.max(am):,.0f}")
                fig = px.histogram(x=am, nbins=60, title="Claim Distribution", color_discrete_sequence=["#f59e0b"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("### 📄 Sample Dataset")
        if st.button("🎲 Generate 5,000 Claims", key="gs"):
            st.session_state["sd"] = generate_sample_dataset(5000)
            st.success("✅ Generated!")
        if "sd" in st.session_state:
            st.dataframe(st.session_state["sd"].head(8), use_container_width=True)
            buf = io.BytesIO()
            st.session_state["sd"].to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            st.download_button("📥 Download Sample", data=buf, file_name="sample_claims.xlsx")

    with e2:
        st.markdown("### 📤 Download Report")
        if "ar" in st.session_state:
            st.success(f"✅ {st.session_state['ar']['n_sim']:,} scenarios ready")
            ebuf = export_report(st.session_state["ar"], st.session_state.get("ap", {}))
            st.download_button("📥 Download Excel Report", data=ebuf, file_name="simulation_report.xlsx", type="primary")
            st.markdown("**Sheets:** Summary, Percentiles, Simulation Data, Parameters")
        else:
            st.warning("⚠️ Run **Aggregate Loss** simulation first, then come here.")

st.markdown("---")
st.caption("Monte Carlo Insurance Simulator | SOA Exam P | Python + Streamlit + Excel")
