# -*- coding: utf-8 -*- 
import numpy as np

if __name__ == '__main__':
    
    arr = np.array([1, 2, 3])
    arr = arr * 5
    print(arr)
    
    arr1 = np.array([[1,2,3],[2,3,4]])
    arr2 = np.array([[3,4,5],[4,5,6]])
    arr = arr1 + arr2
    print(arr)
    
    # 転置
    arr1 = np.array([[1,2,3],[2,3,4]])
    arr2 = arr1.T
    print(arr1)
    print(arr2)