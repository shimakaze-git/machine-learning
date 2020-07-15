import math
import pandas as pd

# ロジスティック回帰分析
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

import matplotlib.pyplot as plt


def logistic(ax):
    ans = 1 / (1 + math.e**-ax)
    return ans


# CSV読み込み
df = pd.read_csv("user_data.csv")

# 目的変数名の指定
y_name = "ユーザ登録"

# 従属変数（使用列）の選択
X_name = ["性別", "学生", "滞在時間"]
X_name = ["学生", "滞在時間"]

# Xとyに分離
X = df[X_name]
y = df[y_name]

model = sm.Logit(y, sm.add_constant(X))
result = model.fit(disp=0)
print(result.summary())

# sklearnの方も使用
lr = LogisticRegression()
lr.fit(X, y)

print("coefficient = ", lr.coef_)  # 係数
print("intercept = ", lr.intercept_)  # 切片

# CSV読み込み
df_test = pd.read_csv("user_data_future.csv")
y_result = []

for i in range(len(df_test.index)):
    y_tmp = result.params.const
    for j in range(len(X_name)):
        x_name = X_name[j]
        y_tmp += result.params[x_name] * df_test[x_name][i]

    y_result.append(logistic(y_tmp))

print(y_result)


# plt.plot(X, y, 'o')
# plt.plot(X, result.params.const+result.params[X_name]*X)
plt.show()
