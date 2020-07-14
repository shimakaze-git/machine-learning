import sklearn.datasets as datasets
# from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
# import statsmodels.formula.api as sm

import pandas as pd


# データーセットからアイリスのデーターセットを読み込む
def iris_datasets():
    iris = datasets.load_iris()

    data_list = []
    target_list = []

    for data, target in zip(iris.data, iris.target):
        if target in [0, 1]:
            data_list.append(list(data))
            target_list.append(target)
    return data_list, target_list

# # 構造を確認
# print(iris.feature_names)

# # 特徴料
# # sepal length (cm), ガクの長さ[cm]
# # sepal width (cm), ガクの幅[cm]
# # petal length (cm) 花弁の長さ[cm]
# # petal width (cm) 花弁の幅[cm]
# print(iris.feature_names)

# # ラベル
# # setosa
# # versicolor
# # virginica
# print(iris.target_names)

# # 特徴量をまとめたデーターを表示する
# print(iris.data)

# 正解ラベルを表示
# print(iris.target)
# print(iris.data)


X, y = iris_datasets()
print(X, y)

X = pd.DataFrame(X)
y = pd.DataFrame(y)

print(X)
# ロジスティック回帰分析
# model = LogisticRegression()
# result = model.fit(X, y)
# print(result)
# print(result.summary())

# sm.add_constant(X))
model = sm.Logit(y, sm.add_constant(X))
r = model.fit(disp=0)

# print(r)
# print('Parameters: ', logit_res.params)
# print logit_res.summary()
