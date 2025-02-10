''' Write a program which will find all such numbers which are divisibale by 7'''
'''but are not a multiple of 5 , between 2000 and 3200(both included)'''
'''the numbers obtained should be printed in a comma-separated
 sequence on a single line '''
 

for i in range(2000,3200):
  if i% 7==0 and i% 5!=0:
    print(i) # vấn đề đây là làm sao có thể in nó trên 1 dòng ngăn cách bằng dấu phẩy