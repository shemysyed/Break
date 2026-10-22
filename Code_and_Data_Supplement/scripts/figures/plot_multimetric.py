from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

results=pd.read_csv("results/nih_main_results.csv")
methods=["Teacher","Standard Retrain","Privacy-Retrain","PULSE"]
frame=results.set_index("method").loc[methods]
metrics=[("forget_auc","Forget AUC $\\uparrow$"),("pair_reid_auc","Pair ReID $\\downarrow$"),("cmc_rank1","CMC-R1 $\\downarrow$"),("map_at_r","mAP@R $\\downarrow$")]
fig,axes=plt.subplots(1,4,figsize=(10,2.7))
for ax,(column,title) in zip(axes,metrics):
 ax.bar(range(len(methods)),frame[column].values,edgecolor="black",linewidth=.5)
 ax.set_title(title); ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods,rotation=35,ha="right",fontsize=7); ax.grid(axis="y",alpha=.25); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); out=Path("figures"); out.mkdir(exist_ok=True); fig.savefig(out/"multimetric_results.pdf",bbox_inches="tight"); fig.savefig(out/"multimetric_results.png",dpi=300,bbox_inches="tight"); print("Saved",out)
