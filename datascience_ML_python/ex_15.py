import numpy as np




data = np.ones((4,6))
print(data.shape)
# trong thực tế thì ta lên cho nó nhiều nghiệm
data = np.reshape(data , (2,-1 , 2 , 2))
print(data.shape)