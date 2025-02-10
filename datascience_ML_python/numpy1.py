import numpy as np
a = np.random.rand(3,4,6)
b = np.random.rand(3,4,6)
c=np.random.rand(1,4,6)# # có thể thực hiện được khi cùng số chiều or chỉ số là 1
print(a)
print("\n")
print(b)
print("\n")
print(a+b)

# quy tắc nếu 2 mảng kích thước giống y tranh nhau , or số chiều kích thước của chúng khác nhau các 
# bé hơn sẽ được quảng bé lên cái lớn hơn 
# có thể thực hiện được khi cùng số chiều or chỉ số là 1
