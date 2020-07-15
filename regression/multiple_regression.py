import statsmodels.api as sm

from sklearn.model_selection import train_test_split
import pandas as pd

# CSV読み込み
data = pd.read_csv(
    'boston_data.csv'
)

data_train, data_test = train_test_split(data)

df = data_train
df_test = data_test

# 目的編数名の指定
y_name = 'PRICE'

# 従属変数（使用例）の選択
X_name = [
    'CRIM',
    'ZN',
    # 'INDUS',
    'CHAS',
    'NOX',
    'RM',
    # 'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT'
]

# Xとyに分離
X = df[X_name]
y = df[y_name]

# 回帰分析
model = sm.OLS(y, sm.add_constant(X))
result = model.fit(disp=0)

print(result.summary())

df_test_data = df_test[X_name]

# 未知のデータの推測
y_result = []

# 未知データの推測（重回帰）
for i in range(len(df_test_data.index)):
    y_tmp = result.params.const

    for j in range(len(X_name)):
        x_name = X_name[j]

        a_i = result.params[x_name]
        x_i = list(df_test_data[x_name])[i]

        y_tmp += a_i * x_i
    y_result.append(y_tmp)

# 推測したデーターと正解データーを比べる
for i in range(len(df_test_data.index)):
    correct = list(df_test['PRICE'])[i]
    y = y_result[i]

    print(abs(correct - y))
