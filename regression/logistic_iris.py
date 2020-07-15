from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

import pandas as pd

# CSV読み込み
df = pd.read_csv("iris.csv")
iris_df = df.query('target != 2')
iris_df.head()

# X = iris_df[['petal length (cm)']]
X = iris_df.drop('target', axis=1)
Y = iris_df['target']

# 80%のデータを学習データに、20%を検証データにする
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# ロジスティック回帰モデルのインスタンスを作成
lr = LogisticRegression()

# ロジスティック回帰モデルの重みを学習
lr.fit(X_train, Y_train)

print("coefficient = ", lr.coef_)
print("intercept = ", lr.intercept_)

Y_pred = lr.predict(X_test)
print(Y_pred)

print('confusion matrix = \n', confusion_matrix(y_true=Y_test, y_pred=Y_pred))
print('accuracy = ', accuracy_score(y_true=Y_test, y_pred=Y_pred))
print('precision = ', precision_score(y_true=Y_test, y_pred=Y_pred))
print('recall = ', recall_score(y_true=Y_test, y_pred=Y_pred))
print('f1 score = ', f1_score(y_true=Y_test, y_pred=Y_pred))
