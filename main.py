import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing

# Read from a csv and set index column
trainDataSet = pd.read_csv("C:/Users/gondolin/PycharmProjects/MLTitanic/DescionTreePythonML/data/train.csv",
                           index_col='PassengerId')
trainDataSet = trainDataSet[trainDataSet.columns.tolist()]
trainDataSet['Sex'] = trainDataSet['Sex'].map({'male': 0, 'female': 1})
trainDataSet['Embarked'] = trainDataSet['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

trainDataSet.dropna()
# remove non needed columns
X = trainDataSet.drop('Survived', axis=1)
# print(X.get('Embarked'))
# X = X.drop('Cabin', axis=1)
# X = X.drop('Cabin', axis=1)
print(X.head(n=0))
y = trainDataSet['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = tree.DecisionTreeClassifier()
# model.fit(X_train, y_train)
