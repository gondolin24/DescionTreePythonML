import pandas as pd
from sklearn import tree

# Read from a csv and set index column
trainDataSet = pd.read_csv("C:/Users/gondolin/PycharmProjects/MLTitanic/DescionTreePythonML/data/train.csv", index_col='PassengerId')

