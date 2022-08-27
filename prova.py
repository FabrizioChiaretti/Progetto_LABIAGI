import numpy as np
from random import randint
import cv2
import glob

def main():
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 700, 700)
    
    for img in glob.glob("Sudoku/Mat1/matrix1.jpg"): #"Sudoku/Mat1/matrix.jpg" 
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    ret,image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    image = cv2.GaussianBlur(image,(5,5),0)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    image = cv2.Canny(image,50,150,apertureSize = 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    lines = cv2.HoughLines(image,1,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
        
    return
    
    
main()