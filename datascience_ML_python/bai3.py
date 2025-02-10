'''find the strogest neighbour. Given an array of N positive integers'''
""" The task is to find maximum for every adjacent pair in the array"""

def maximumAdjacent(arr1,n):
    arr2 =[]
    for i in range(1,n):
        r = max(arr1[i], arr1[i-1])
        arr2.append(r)
    for ele in arr2:
        print(ele,end=" ")
data3=[4,5,6,7,3,9,11,2,10]

n = len(data3)
maximumAdjacent(data3,n)





  

#take care of  that  problem
#key-word:"take care of"
