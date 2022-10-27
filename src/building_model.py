import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
X = df[['Fare', 'Age']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)

print(model.coef_, model.intercept_)
# [[ 0.01615949 -0.01549065]] [-0.51037152]