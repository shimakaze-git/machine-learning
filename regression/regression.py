# 回帰分析

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import matplotlib.pyplot as plt

import pandas as pd

# CSV読み込み
df = pd.read_csv('iris.csv')
iris_df = df.query('target != 2')
iris_df.head()

X = iris_df[['petal length (cm)']]
X = iris_df.drop('target', axis=1)
Y = iris_df['target']

# 80%のデータを学習データに、20%を検証データにする
X_train, X_test, Y_train, _ = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# 回帰分析の実行

model = sm.OLS(Y_train, sm.add_constant(X_train))
result = model.fit(disp=0)
print(result.summary())

y_result = []
for i in range(len(X_test.index)):
    b = float(result.params.const)
    a_i = float(result.params[['petal length (cm)']])
    x_i = list(X_test['petal length (cm)'])[i]

    y_tmp = b + a_i * x_i
    # print(y_tmp)
    y_result.append(y_tmp)

# print(len(X_test), len(y_result))
print(y_result)

# plt.plot(X_test, y_result, 'o')
# plt.plot(X_test, y_result)
# plt.plot(X_train, Y_train)
# plt.show()
