import pandas as pd
import numpy as np 

element = pd.read_csv("element_data.csv", index_col="Element")
unique = pd.read_csv("unique_m.csv")

tc = unique["critical_temp"]
mat = unique["material"]

unique.drop(columns=["critical_temp", "material"], inplace=True)
print(element.columns)
element.drop(columns =["SuperconductingPoint"], inplace=True)
for name in ["Block","CrystalStructure", "MagneticType", "Phase", "Series" ]:
    #one hot encoding for str columns
    element = pd.concat([element, pd.get_dummies(element[name], dtype = np.int)], axis = 1)
    element.drop(columns =[name], inplace=True)

unweighted_data = unique.dot(element)

unweighted_data["critical_temp "] = tc
unweighted_data["material"] = mat
unweighted_data.to_csv("unweighted_data.csv")