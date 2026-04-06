import numpy as np, pandas as pd
from datetime import datetime, timedelta

def generate_sample_dataset(n=5000, seed=42):
    np.random.seed(seed)
    start = datetime(2022,1,1)
    days = (datetime(2024,12,31) - start).days
    dates = [start + timedelta(days=int(d)) for d in np.random.randint(0, days, n)]
    settle = [d + timedelta(days=int(s)) for d, s in zip(dates, np.random.randint(30, 180, n))]
    types = np.random.choice(["Auto","Property","Liability"], n, p=[0.5,0.3,0.2])
    amounts = np.array([np.random.lognormal({"Auto":7.0,"Property":7.8,"Liability":8.2}[t],
                        {"Auto":1.0,"Property":1.3,"Liability":1.5}[t]) for t in types]).round(2)
    npol = int(n * 0.7)
    pols = [f"POL-{np.random.randint(10000,99999)}" for _ in range(npol)]
    deducts = np.random.choice([250,500,1000,2000], n, p=[0.2,0.4,0.3,0.1])
    return pd.DataFrame({
        "Claim_ID": [f"CLM-{str(i+1).zfill(5)}" for i in range(n)],
        "Policy_ID": np.random.choice(pols, n),
        "Date_Filed": dates, "Date_Settled": settle, "Claim_Type": types,
        "Claim_Amount": amounts, "Deductible": deducts,
        "Net_Claim": np.maximum(amounts - deducts, 0).round(2),
        "Region": np.random.choice(["North","South","East","West"], n, p=[.25,.3,.2,.25]),
        "Age_Group": np.random.choice(["18-25","26-40","41-60","60+"], n, p=[.15,.35,.35,.15]),
        "Risk_Class": np.random.choice(["Low","Medium","High"], n, p=[.5,.35,.15]),
        "Status": np.random.choice(["Settled","Open","Denied"], n, p=[.8,.12,.08]),
    }).sort_values("Date_Filed").reset_index(drop=True)
