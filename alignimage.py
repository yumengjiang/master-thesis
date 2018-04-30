import cv2  
import os  
      
import numpy as np 
print("sign4") 
#root_path = "home/project/master-thesis/results_perfect/" 
dir = os.sep+"images"  
print("sign3")
count = 0  
for root,dir,files in os.walk(dir): 
    print("sign5") 
    for file in files:  
        print("sign1")
        srcImg = cv2.imread(os.sep+"images"+"/"+str(file))  
        roiImg = srcImg[1:180, 1:640]  
        print("sign2")
        cv2.imwrite(os.sep+"Image"+"/"+str(file),roiImg)  
        count +=1  
        if count%400==0:  
            print (count)  