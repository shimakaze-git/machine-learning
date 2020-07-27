import numpy as np
from similarity import _similarity, _pdist
from scipy.spatial.distance import squareform


x_list = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1]
]
x = np.array(x_list)
# print(x)
# print(x.T)

item1 = x.T[0]
item2 = x.T[1]
sim = _similarity(item1, item2)

x_t = x.T

# print(sim)

print(x)
print(x_t)

# https://ja.wikipedia.org/wiki/%E8%B7%9D%E9%9B%A2%E8%A1%8C%E5%88%97
# 距離行列を算出
# xの行数がNの場合はN*(N-1)/2
d = _pdist(x_t, 'cosine')
# d = _pdist(x, 'cosine')

d = 1 - d
print(d)
print(len(d))
print(type(d))

# item1 = x_t[1]
# item2 = x_t[4]
# print(item1, item2)
# sim = _similarity(item1, item2)
# print(sim)

d_ = squareform(d)
print(d_)

me = np.eye(d_.shape[0])
# print(d_.shape[0])
# print(d_.shape)
# print(me)
d_ = d_ - me
print(d_)

# item1〜6まで、類似度の高い順に３つ推薦対象とする
for idx in range(x_t.shape[0]):
    print('item {}'.format(idx+1))

    item1_sim = d_[:, idx]

    item1_rel = []
    for i in range(item1_sim.shape[0]):
        item1_rel.append(
            ('item{}'.format(i+1), item1_sim[i])
        )

    # print(item1_rel, '||')
    item1_rel = sorted(item1_rel, key=lambda d_: d_[1], reverse=True)
    # print(item1_rel)

    # 3件に絞る
    item1_rel = item1_rel[:3]
    print("   ", item1_rel)
    print("")
