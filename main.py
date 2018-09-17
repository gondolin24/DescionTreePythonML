import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

testDataSet = pd.read_csv("C:/Users/gondolin/PycharmProjects/MLTitanic/DescionTreePythonML/data/test.csv",
                           index_col='PassengerId')
# Read from a csv and set index column
trainDataSet = pd.read_csv("C:/Users/gondolin/PycharmProjects/MLTitanic/DescionTreePythonML/data/train.csv",
                           index_col='PassengerId')
trainDataSet = trainDataSet[trainDataSet.columns.tolist()]


#Setup Test Data
testDataSet['Sex'] = testDataSet['Sex'].map({'male': 0, 'female': 1})
testDataSet['Embarked'] = testDataSet['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
yy = testDataSet
yy = yy.drop('SibSp', axis=1)  # not needed
yy = yy.drop('Parch', axis=1)  # not needed
yy = yy.drop('Embarked', axis=1)  # not needed
yy = yy.drop('Fare', axis=1)  # not needed
yy = yy.drop('Age', axis=1)  # not needed
X_test = yy
y_train = trainDataSet.get('Survived')
# Data Sanataize
trainDataSet['Sex'] = trainDataSet['Sex'].map({'male': 0, 'female': 1})
trainDataSet['Embarked'] = trainDataSet['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
trainDataSet.dropna()

X = trainDataSet.drop('Survived', axis=1)
X = X.drop('Name', axis=1)  # not needed
X = X.drop('Ticket', axis=1)  # not needed
X = X.drop('Cabin', axis=1)  # not needed

AgeCol = X.get('Age')
intCount = 0
intSum = 0

for passAge in AgeCol:
    if ~np.isnan(passAge):
        intSum = passAge + intSum
        intCount = intCount + 1
averageAge = intSum/intCount
# There has to be a better way

X.get('Age').fillna(averageAge, inplace=True)

# trainDataSet['Age'] = trainDataSet['Age'].map({'nan': averageAge})
X = X.drop('SibSp', axis=1)  # not needed
X = X.drop('Parch', axis=1)  # not needed
X = X.drop('Embarked', axis=1)  # not needed
X = X.drop('Fare', axis=1)  # not needed

print(X.head(n=0))
y = trainDataSet['Survived']
X_train = X

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))