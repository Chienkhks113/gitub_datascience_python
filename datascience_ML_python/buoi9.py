''' Ex1 Write a Numpy program to reverse an array ( first element becomes last)


'''
minh=[]
tuan = [12,13,14 ,15 ,16, 17, 18, 19,20]
# Đảo ngược list bằng hàm reversed() và chuyển đổi thành list
# Cách 2: Sử dụng hàm reversed()
minh = list(reversed(tuan))
print(minh)


'''
Write a Numpy program to test whether each element of a 1D array
is also present in a second array

'''

import numpy as np
array1 = np.array([0,10,20,40,60])
array2 =np.array([10,30,40])
result = np.in1d(array1,array2)
print(result)
'''
 Write a Numpy program to find the indices of the maximum and minimum values along the given axis of an array
'''
arr = np.array([1,6,4,8,9,-4,-2,11])
max_index = np.argmax(arr)
min_index = np.argmin(arr)
print(max_index)
print(min_index)
