"""
find best values from SVM and XGB
"""
from linear_toi_meta import best_par
import pandas as pd 
from pathlib import Path



p = Path('./SVM')
df = pd.read_csv(p/'toi2.csv')
idx = df[df["data set"]=='eval']["MSE"].idxmin()
best = df.iloc[idx]
best_par(best, p)

p = Path('./XGB')
for name in ['gain', 'weights']:
    f = 'toi_%s.csv'%name
    df = pd.read_csv(p/f)
    idx = df[df["data set"]=='eval']["MSE"].idxmin()
    best = df.iloc[idx]
    best_par(best, p)