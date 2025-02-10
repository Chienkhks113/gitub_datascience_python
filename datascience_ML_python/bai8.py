'''
 Let user 2 words in English as input. Print out the output
 # which is the shortest chain according to the following rules:
 # each word in the chain has at least 3 letters
 # the 2 input words from user will be used as the first and the last
 words of the chain
 # 2 last letters of 1 word will be the same as 2 first letters of the next word in the chain
 # all the words are from the file wordsEn.txt
 # if there are multiple shortest chains, return any of them is
 sufficient
'''
''' i need to make an effort more than. the success is comming with us. if we take a chance it'''
import pandas as pd
import random
def wordsEn():
    data = pd.read_csv("wordsEn.txt")
    # print(len(data.values))
    lt = []
    for i in range(len(data)):
        word = str(data.values[i][0])
        if len(word) >= 3:
            lt.append(word)
    return lt

def find_chain(beginword,endword,words):
    chain = [beginword]
    while beginword[-1] != endword[0]:
        list_temp = []
        for i in range(len(words)):
            if beginword[-1] == words[i][0]:
                list_temp.append(words[i])
        beginword = list_temp.pop(random.randrange(len(list_temp)))
        chain.append(beginword)

    chain.append(endword)
    return chain


words = wordsEn()

beginword=str(input("enter begin word (>3 letters): "))
endword=str(input("enter end word (>3 letters): "))

# beginword="father"
# endword="mother"

chain=find_chain(beginword,endword,words)

for i in range(100):
    test=find_chain(beginword,endword,words)
    if len(test)<len(chain):
        chain=test

print(chain)