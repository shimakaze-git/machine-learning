import numpy as np

from scipy.spatial.distance import correlation, cosine, pdist
from scipy import stats


# cos類似度
def cos_similarity(item1, item2):
    return 1 - cosine(item1, item2)


# Pearsonの(積率)相関係数
def pearson_corr(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)

    # 分子 / 分母
    numerator = np.dot(x_diff, y_diff)
    denominator = (np.sqrt(sum(x_diff ** 2)) * np.sqrt(sum(y_diff ** 2)))

    return numerator / denominator


# scipyによるPearsonの(積率)相関係数
def distance_correlation(x, y):
    return 1 - correlation(x, y)


# p値も算出する
def stats_pearsonr(x, y):
    return stats.pearsonr(x, y)


# item1とitem2の類似度を算出する
def _similarity(item1, item2):
    similarity_value = cos_similarity(item1, item2)
    return similarity_value


# 距離行列を算出する
# xの行数がNの場合はN*(N-1)/2
def _pdist(x, method):
    d = pdist(x, 'cosine')
    return d
