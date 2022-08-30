
import numpy as np
from random import randint
import cv2
import glob
import pandas as pd
import pytesseract
import tensorflow as tf

def main():
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 700, 700)
    
    for img in glob.glob("Sudoku/Mat2/matrix.jpg"): 
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    ret,image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)
    
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    x, y, w, h = cv2.boundingRect(cnts[1])
    matrix = image[y:y+h, x:x+w]
    cv2.imshow("image", matrix)
    cv2.waitKey(0)
    
    dimensions = matrix.shape
    cnt = 0
    w = h = dimensions[0] // 9
    
    mat = [[0 for i in range(0,9)] for j in range(0,9)]
    
    for y in range(0, 9):
        y1 = y*h
        for x in range(0, 9):
            x1 = x*w
            cell = matrix[y1+10:y1+h-10, x1+10:x1+w-10]
            cv2.imshow("image", cell)
            cv2.waitKey(0)
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            n = pytesseract.image_to_string(cell, config = custom_config).split()
            if len(n) > 0 :
                print(n[0])
                mat[y][x] = int(n[0])
            else:
                mat[y][x] = 0
            cnt += 1
            
    cv2.destroyAllWindows()
            
    if (cnt != 81):
        print("I did not recognise cells or numbers, try again")
        return 
    
    else:
        print("Matrix detection completed")
        
    sudoku_matrix = np.array(mat, dtype = np.int32)
    print(sudoku_matrix)
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)
    
    # create a sequential neural network
    #model = tf.keras.models.Sequential() 
    # add layers to model
    #model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
    #model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    #model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    #model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    
    #model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    #model.fit(x_train, y_train, epochs = 3)
    #model.save('Digits.model')
    
    model = tf.keras.models.load_model('Digits.model')
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss: ', loss, 'accuracy: ', accuracy)
    
    #image = cv2.GaussianBlur(image,(5,5),0)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #image = cv2.Laplacian(image,cv2.CV_32F)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #roi = cv2.selectROI(image)
    #roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #cv2.imshow("roi", roi_cropped)
    #cv2.waitKey(0)
    
    #roi_number = 0
    #for idx in range(len(cnts)):
        #roi_number += 1
        #x, y, w, h = cv2.boundingRect(cnts[idx])
        #roi = image[y:y+h, x:x+w]
        #cv2.imshow("image", roi)
        #cv2.waitKey(0)
    #print(roi_number)   
    
    #ROI_number = 0
    #for i in range(len(cnts)):
        #if (i % 2 == 0):
            #cnt = cnts[i]
            #x,y,w,h = cv2.boundingRect(cnt)
            #ROI = image[y:y+h, x:x+w]
            #cv2.imshow("ROI", ROI)
            #cv2.waitKey(0)
            #ROI_number += 1

    #print(ROI_number) #dovrebbe essere uguale a 9x9=81, invece è uguale a 79
    
    #image = cv2.Canny(image,100,200)
    #cv2.imshow("image", image)
    #cv2.waitKey(0) 
    #image = cv2.GaussianBlur(image,(5,5),0)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #image = cv2.Laplacian(image,cv2.CV_64F)
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    #roi = cv2.selectROI(image)
    #roi_cropped = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #cv2.imshow("roi cropped", roi_cropped)
    #cv2.waitKey(0)
    #sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #sharpen = cv2.filter2D(roi_cropped, -1, sharpen_kernel)
    #cv2.imshow('sharpen', sharpen)
    #cv2.waitKey()
    #image = cv2.GaussianBlur(img_thresh, (3,3), 0)
    #kernel = np.ones((5, 5), np.uint8)
    #image = cv2.erode(blur_img, kernel) 
    #cv2.imshow("blur_img", image)
    #cv2.waitKey(0)
    #text = pytesseract.image_to_string(sharpen)
    #print(text)
    
    
    return
    
    
    
    
main()