'''Given two matrices(2 nested lists) , the task is to write a python promgram'''
''' to add elements to each row from initial matrix'''

import numpy as np
#given matrices
data5_list1=[[4,3,5],[1,2,3],[3,7,4]]
data5_list2=[[1,3],[9,3,5,7],[8]]
#convert lists to NumPy arrays
arr1 = np.array(data5_list1)
arr2 = np.array(data5_list2)
# Add elements to each row
result=np.hstack((arr1,arr2))
print(result)

1 # I know a new method well . It is used in matrices   
# for i, j in zip(data5_list1, data5_list2): my_matrix.append(i+j):  Zip connects these ojects of matrices