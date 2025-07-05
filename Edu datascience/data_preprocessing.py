import pandas as pd
data = pd.read_csv('dataset_med.csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace = True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[data.select_dtypes(include='number').columns] = scaler.fit_transform(data.select_dtypes(include='number'))

data.to_csv("cleaned_test.csv", index=False)

print("Cleaned dataset has been saved to 'cleaned_disease_test.csv'")