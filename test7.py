import pandas as pd 

df = pd.read_csv('dashride_columbia_dataset.csv', sep = ',')

print(df['endLoc'].value_counts())
print(df['startLoc'].value_counts())