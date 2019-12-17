import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv("train.csv")

plt.figure(figsize=(10,10))
plt.hist(df["critical_temp"],50)
plt.xlabel("T$_c$ in K", fontsize=28)
plt.xlim(0,150)
plt.ylabel("Count", fontsize =28)
plt.tick_params(size=22, labelsize=24)
plt.tight_layout()
plt.savefig("hist_tc.pdf")