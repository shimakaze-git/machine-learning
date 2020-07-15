# regression (回帰)

- 単回帰分析
- 重回帰分析(multiple regression)

# 数式

# data

- boston_data.csv ボストンの住宅価格
- breast_cancer.csv がんの診断結果

csv ファイルの生成

```
$ python
>>> import numpy as np
>>> import pandas as pd

>>> from sklearn.datasets import load_boston
>>> boston = load_boston()
>>> boston_df = pd.DataFrame(boston.data)
>>> boston_df['PRICE'] = pd.DataFrame(boston.target)
>>> boston.feature_names = np.append(boston.feature_names, 'PRICE')
>>> boston_df.columns = boston.feature_names
>>> boston_df.to_csv("boston_data.csv")

>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> iris_df = pd.DataFrame(iris.data)
>>> iris_df['target'] = pd.DataFrame(iris.target)
>>> iris.feature_names = np.append(iris.feature_names, 'target')
>>> iris_df.columns = iris.feature_names
>>> iris_df.to_csv('iris.csv')
```

# Links

- [Python によるロジスティック回帰分析](https://analysis-navi.com/?p=1229)
- [Python による重回帰分析](https://analysis-navi.com/?p=1930)
