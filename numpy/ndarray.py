import numpy as np

arr = np.array([1,2,3])
print(arr)
print(arr.flags)
print(arr.ndim)
print(arr.size)
print(arr.shape)
print(arr.itemsize)
print(arr.strides)
print(arr.nbytes)
print(arr.dtype)
print("--------------------------------------")

matrix = np.array([[1,2,3],[4,5,6]])
print(matrix)
print(matrix.flags)
print(matrix.ndim)
print(matrix.size)
print(matrix.shape)
print(matrix.itemsize)
print(matrix.strides)
print(matrix.nbytes)
print(matrix.dtype)
print("--------------------------------------")

arr_random = np.random.rand(2, 4)
print(arr_random)
print("--------------------------------------")
