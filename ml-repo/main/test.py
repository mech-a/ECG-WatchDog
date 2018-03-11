import os
import pandas
import matplotlib

x = os.path.join("datasets", "arrythmia-ds.csv")
print(x)
pobj = pandas.read_csv(x)
pobj.info()