from pathlib import Path
import pandas as pd
for name in ("nih_main_results.csv","covidx_results.csv","nih_operating_points.csv","fixed_k_full_metrics.csv"):
    path=Path("results")/name
    print("\n",path)
    print(pd.read_csv(path).to_string(index=False))
