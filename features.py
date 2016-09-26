
import os
import sys

firstfile = open("C:/shruti/project/inputFile_landmarks2.txt")
secondfile = open("C:/shruti/project/inputFile_landmarks1.txt")

firstfileContents = firstfile.readlines()
lastfileContents = secondfile.readlines()
index1 =0
row=""
for f in firstfileContents:
    fvalue = f.strip()
    lvalue = lastfileContents[index1].strip()
    values1 = fvalue.split(" ")
    values2 = lvalue.split(" ")
    index1 = index1 + 1
    # print(float(values1[1]))
    xfeat = float(values1[0]) - float(values2[0])
    yfeat = float(values1[1]) - float(values2[1])
    #print(str(xfeat)+","+str(yfeat))
    row += str(xfeat)+","+str(yfeat)+","
print row