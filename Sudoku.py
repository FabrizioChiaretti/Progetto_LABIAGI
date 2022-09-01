

import numpy as np
import cv2 
import glob
import pytesseract
from time import sleep
from Genetic_algotithm import genetic_algorithm



def main ():
    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 700, 700)
    
    for img in glob.glob("Sudoku/Mat4/matrix.jpg"): 
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    ret,image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    x, y, w, h = cv2.boundingRect(cnts[1])
    matrix = image[y:y+h, x:x+w]
    cv2.imshow("Image", matrix)
    cv2.waitKey(0)
    
    dimensions = matrix.shape
    cnt = 0
    w = h = dimensions[0] // 9
    
    mat = [[0 for i in range(0,9)] for j in range(0,9)]
    
    print('Detecting matrix...')
    
    for y in range(0, 9):
        y1 = y*h
        for x in range(0, 9):
            x1 = x*w
            cell = matrix[y1+10:y1+h-10, x1+10:x1+w-10]
            #cv2.imshow("Image", cell)
            #cv2.waitKey(0)
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            n = pytesseract.image_to_string(cell, config = custom_config).split()
            if len(n) > 0 :
                print(n[0])
                mat[y][x] = int(n[0])
            cnt += 1
            
    cv2.destroyAllWindows()
            
    if (cnt != 81):
        print("I did not recognise cells or numbers, try again")
        return 
    
    else:
        print("Matrix detection completed")
    
    sudoku_matrix = np.array(mat, dtype = np.int32)
    print('Sudoku matrix:\n')
    print(sudoku_matrix)
    
    fit = 1
    cnt = 0
    while fit != 0:
        cnt += 1
        print(f"Attempt number {cnt}")
        sleep(2)
        solution = genetic_algorithm(sudoku_matrix)
        fit = solution.fitness

    print(solution.matrix)
    print(f"Solution found in {cnt} attempts")
    
    return
    
    
    
main()



