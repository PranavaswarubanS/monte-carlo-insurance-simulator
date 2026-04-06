import io, numpy as np, pandas as pd

def read_claims_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)
    col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["amount", "loss", "cost", "severity", "claim_amount"]):
            col = c
            break
    if not col:
        nums = df.select_dtypes(include=[np.number]).columns
        col = nums[0] if len(nums) > 0 else None
    if not col:
        raise ValueError("No numeric column found!")
    amounts = df[col].dropna().values.astype(float)
    return df, amounts[amounts > 0], col

def export_report(result, params=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary = pd.DataFrame({
            "Metric": ["Simulations", "Mean Loss", "Std Dev", "Median", "Skewness", "Min", "Max",
                       "Avg Claims/Year", "VaR 90%", "VaR 95%", "VaR 99%", "TVaR 90%", "TVaR 95%", "TVaR 99%"],
            "Value": [result["n_sim"], round(result["mean"],2), round(result["std"],2),
                     round(result["median"],2), round(result["skew"],4), round(result["min"],2), round(result["max"],2),
                     round(result["mean_claims"],2),
                     round(result["risk"]["VaR_0.9"],2), round(result["risk"]["VaR_0.95"],2), round(result["risk"]["VaR_0.99"],2),
                     round(result["risk"]["TVaR_0.9"],2), round(result["risk"]["TVaR_0.95"],2), round(result["risk"]["TVaR_0.99"],2)]
        })
        summary.to_excel(w, sheet_name="Summary", index=False)

        pcts = [1,5,10,25,50,75,90,95,97.5,99,99.5]
        pct_df = pd.DataFrame({
            "Percentile": [f"{p}th" for p in pcts],
            "Loss": [round(float(np.percentile(result["losses"], p)),2) for p in pcts]
        })
        pct_df.to_excel(w, sheet_name="Percentiles", index=False)

        n = min(5000, len(result["losses"]))
        sim_df = pd.DataFrame({
            "Scenario": range(1, n+1),
            "Claims": result["counts"][:n].astype(int),
            "Total_Loss": np.round(result["losses"][:n], 2)
        })
        sim_df.to_excel(w, sheet_name="Simulation Data", index=False)

        if params:
            params_df = pd.DataFrame({
                "Param": list(params.keys()),
                "Value": [str(v) for v in params.values()]
            })
            params_df.to_excel(w, sheet_name="Parameters", index=False)

    buf.seek(0)
    return buf
