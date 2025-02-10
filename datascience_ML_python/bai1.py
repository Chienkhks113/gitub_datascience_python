''' Write a program to count positive and negative numbers in a list'''

data1 =[-10,-21,-4,-45,-66,93,11,-4,-6,12,11,4]
tuan= 0
minh = 0
for i in data1:
    if i > 0:
     tuan=tuan+1
    else: minh=minh+1

print(tuan)
print(minh)