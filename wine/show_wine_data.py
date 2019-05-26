import pandas as pd
from sklearn.datasets import load_wine

# ワインのデータをインポート
data = load_wine()
# データの概要を表示
# print(data.DESCR)

# Pandasを用いて特徴量とカラム名を取り出す
data_x = pd.DataFrame(
    data=data.data,
    columns=data.feature_names
)

print(data.data)
print(data.feature_names)

# データが持つ特徴量を上から5行表示
print(data_x)
print(data_x.head())

# Pandasを用いてラベルを取り出す
data_y = pd.DataFrame(data=data.target)
data_y = data_y.rename(columns={0: 'class'})

print(data.target)
# print(data_y)
# print(data_y.head())
