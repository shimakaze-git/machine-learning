# -*- coding: utf-8 -*- 
import numpy as np

if __name__ == '__main__':
    
    arr = np.array([1, 2, 3])
    arr = arr * 5
    print(arr)

    # 加算
    arr1 = np.array([[1,2,3],[2,3,4]])
    arr2 = np.array([[3,4,5],[4,5,6]])
    arr = arr1 + arr2
    print(arr)
    arr = np.add(arr1,arr2)
    print(arr)
    
    
    A = np.array([[3,7],[6,4]])
    B = np.array([[0,3],[4,4]])
    arr = np.add(A,B)
    print(arr)
    
    
    
    
    
    # 転置
    arr1 = np.array([[1,2,3],[2,3,4]])
    arr2 = arr1.T
    print(arr1)
    print(arr2)
    
    # https://oguemon.com/study/linear-algebra/matrix-op/