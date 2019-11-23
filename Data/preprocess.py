import pandas as pd

df = pd.read_csv("raw.csv")
#drop empty cols + unnamed cols
l_befor = len(df.columns)
df = df.dropna(axis='columns', how="all")
df = df. drop(columns=["Unnamed: 0",	"Unnamed: 0.1",	"Unnamed: 0.1.1",	"Unnamed: 0.1.1.1", "Unnamed: 0.1.1.1.1" ])
print("befor: ", l_befor,", after: ", len(df.columns))

#group by num colum
l_befor = len(df.index)
df=df.groupby("num").first(dropna=True)
print("befor: ", l_befor,", after: ", len(df.index))

df.to_csv("grouped_raw_data.csv")