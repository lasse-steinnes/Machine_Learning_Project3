from pathlib import Path
import pandas as pd
import os

data_path = Path("./Data/")

for files in data_path.glob('*.csv'):
    print(pd.read_csv(files))

