import cv2  
import os  
      
import numpy as np 

#root_path = "home/project/master-thesis/results_perfect/" 
dir = "images"  

count = 0  
for root,dir,files in os.walk(dir): 
     
    for file in files:  
        
        srcImg = cv2.imread("images"+"/"+str(file))  
        roiImg = srcImg[1:260, 1:640]  
        
        cv2.imwrite("Image"+"/"+str(file),roiImg)  
        count +=1  
        if count%400==0:  
            print (count)  