
import pandas as pd

'''
简单测试读写
'''
file1  = '.\\stanford_campus_dataset\\annotations\\deathCircle\\video0\\annotations.txt'
f1 = open(file1)
print("Name of the file:", f1.name)
line = f1.readline()
print("Read Line: %s" % (line))

'''
简单测试读写
'''
