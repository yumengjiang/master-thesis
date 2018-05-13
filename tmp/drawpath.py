import cv2  
import numpy
import matplotlib.pyplot as plt 
import os  
      
 

a = numpy.loadtxt('centers-all.txt') 
A=a[:,0]
B=a[:,2]
fig = plt.figure()  
ax1 = fig.add_subplot(111) 
ax1.scatter(A,B,c = 'r',marker = '.')  
#plt.scatter(A,B)
print(len(a))
plt.show()   