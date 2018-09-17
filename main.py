import pandas as pd
from sklearn.model_selection import train_test_split

# Read from a csv and set index column
trainDataSet = pd.read_csv("C:/Users/gondolin/PycharmProjects/MLTitanic/DescionTreePythonML/data/train.csv",
                           index_col='PassengerId')
trainDataSet = trainDataSet[trainDataSet.columns.tolist()]
trainDataSet['Sex'] = trainDataSet['Sex'].map({'male': 0, 'female': 1})
