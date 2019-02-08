# conding: utf-8

import sklearn.datasets as datasets

# データーセットからアイリスのデーターセットを読み込む
iris = datasets.load_iris()

# 構造を確認
print(iris.feature_names)

# 特徴料
# sepal length (cm), ガクの長さ[cm]
# sepal width (cm), ガクの幅[cm]
# petal length (cm) 花弁の長さ[cm]
# petal width (cm) 花弁の幅[cm]
print(iris.feature_names)

# ラベル
# setosa
# versicolor
# virginica
print(iris.target_names)

# 特徴量をまとめたデーターを表示する
print(iris.data)

# 正解ラベルを表示
print(iris.target)
