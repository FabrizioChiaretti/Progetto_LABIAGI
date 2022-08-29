
import numpy as np
from random import randint
from time import sleep
import cv2 
import glob
import tensorflow as tf


class chromosome:
    
    def __init__(self, mat):
        self.matrix = np.copy(mat)
        self.fitness = None
        


def best_fitness_left(population):
    result = None
    best_fitness = 800
    n = len(population)
    
    for i in range(0, n):
        chrom = population[i]
        if chrom.fitness < best_fitness:
             best_fitness = chrom.fitness
             result = chrom
    
    return result



def best_fitness_right(population):
    result = None
    best_fitness = 800
    n = len(population)
       
    for i in range(n-1, -1, -1):
        chrom = population[i]
        if chrom.fitness < best_fitness:
             best_fitness = chrom.fitness
             result = chrom
    
    return result



def fitness(chrom):
    
    matrix = chrom.matrix
    value = 0
    
    for i in range(0,9):
        for j in range(0,9):
            num = matrix[i][j]
            for k in range(j+1, 9):
                if matrix[i][k] == num:
                    value += 1
            for k in range(i+1, 9):
                if matrix[k][j] == num:
                    value += 1
            
    return value



def check_grid(matrix, value_list, i, j):
    
    for r in range(i, i+3):
        for c in range(j, j+3):
            if (matrix[r][c] != 0):
                value_list.remove(matrix[r][c])
    
    return



def genetic_algorithm(mat):
    
    print('Genetic algorithm starts')
    result = None
    
    # initialize population
    population = list()
    population_dim = 600
    for i in range(0,population_dim):
        chrom = chromosome(mat)
        i = 0
        j = 0
        for k in range(1, 10):
            value_list = [i for i in range(1,10)]
            check_grid(chrom.matrix, value_list, i, j)
            for r in range(i, i+3):
                for c in range(j, j+3):
                  if chrom.matrix[r][c] == 0:
                    pos = randint(0, len(value_list)-1)
                    value = value_list[pos]
                    value_list.remove(value)
                    chrom.matrix[r][c] = value
                    
            j += 3
            if k % 3 == 0:
                i += 3
                j = 0
            
        chrom.fitness = fitness(chrom)
        population.append(chrom)  
    
    c = best_fitness_left(population)
    prevfit = c.fitness
    cnt = 0
    
    while (cnt < 500):
        
        print('Iteration: ', cnt)
        # stopping criterion
        result = best_fitness_left(population)
        bestfit = result.fitness
        print('population_dim: ', len(population))
        print('update: ', prevfit - bestfit)
        print('best fitness: ', bestfit)
        if result.fitness == 0:
            print('Solution found')
            break
        
        # selection
        sub = list()
        sub_dim = len(population) // 2
        if sub_dim % 2:
            sub_dim += 1
        
        for i in range(0, sub_dim):
            if i % 2:
                best = best_fitness_left(population)
            else:
                best = best_fitness_right(population)
            sub.append(best)
            population.remove(best)
    
        # crossover
        for i in range(0, len(sub) // 2):
            num1 = randint(0, len(sub)-1)
            parent1 = sub[num1]
            population.append(sub[num1])
            sub.remove(sub[num1])
            num2 = randint(0, len(sub)-1)
            parent2 = sub[num2]
            population.insert(0, sub[num2])
            sub.remove(sub[num2])
            matrix = []
            for i in range(0,3):
                matrix.append(parent1.matrix[i])
                
            for i in range(3,6):
                matrix.append([0,0,0,0,0,0,0,0,0])
                for j in range(0,9):
                    if (j <= 5):
                        matrix[i][j] = parent1.matrix[i][j]
                    else:
                        matrix[i][j] = parent2.matrix[i][j]
            
            for i in range(6,9):
                matrix.append(parent2.matrix[i])
            
            m = np.array(matrix)
            son = chromosome(m)
            son.fitness = fitness(son)
            if (i % 2):
                population.append(son)
            else:
                population.insert(0, son)
            
        population_dim = len(population)
         
        # new popolation
        population_dim -= (sub_dim // 2)
        new_population = list()
        for i in range(0, population_dim):
            if i % 2: 
                chrom = best_fitness_left(population)
            else:
                chrom = best_fitness_right(population)
            new_population.append(chrom)
            population.remove(chrom)

        population = new_population
            
        # mutation
        for k in range(0, len(population)):
            num = randint(1, 100)
            chrom = population[k]
            
            if (num <= 10):
                grid = randint(1,9)
                if grid <= 3:
                    i = 0
                elif grid <= 6:
                    i = 3 
                else: 
                    i = 6
                
                if grid == 1 or grid == 4 or grid == 7:
                    j = 0
                elif grid == 2 or grid == 5 or grid == 8:
                    j = 3
                else:
                    j = 6 
                
                v1 = 1
                r1 = 0
                r2 = 0
                c1 = 0
                c2 = 0
                while v1 != 0:
                    r1 = randint(i, i+2)
                    c1 = randint(j, j+2)
                    v1 = mat[r1][c1]
                   
                v2 = 1
                while (v2 != 0) or (r1 == r2 and c1 == c2):
                    r2 = randint(i, i+2)
                    c2 = randint(j, j+2)
                    v2 = mat[r2][c2]
                 
                aux = chrom.matrix[r1][c1]
                chrom.matrix[r1][c1] = chrom.matrix[r2][c2]
                chrom.matrix[r2][c2] = aux   
                chrom.fitness = fitness(chrom)  
                
        prevfit = bestfit
        cnt += 1
        #sleep(1)
    
    print('Genetic algorithm finished')
    fit = result.fitness
    cnt = 0
    for c in population:
        if (c.fitness == fit):
            cnt += 1
    
    print(cnt)
    return result



def main ():
    
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
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 700, 700)
    
    for img in glob.glob("Sudoku/Mat2/matrix.jpg"): 
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
    ret,image = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    
    cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, cnts, -1, (0,255,0), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    
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
            cell = matrix[y1:y1+h, x1:x1+w]
            cv2.imshow("image", cell)
            cv2.waitKey(0)
            cell = cv2.resize(cell, (28,28))
            cv2.imshow("image", cell)
            cv2.waitKey(0)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(cell, -1, sharpen_kernel)
            cv2.imshow('image', sharpen)
            cv2.waitKey()
            img = np.array([cell])
            img = np.invert(img)
            prediction = model.predict(img)
            print('number detected: ', np.argmax(prediction))
            mat[y][x] = np.argmax(prediction)
            cnt += 1
            
    cv2.destroyAllWindows()
            
    if (cnt != 81):
        print("I did not recognise cells or numbers, try again")
        return 
    
    else:
        print("Matrix detection completed")
    
    sudoku_matrix = np.array(mat, dtype = np.int32)
    print(sudoku_matrix)
    
    #fit = 1
    #cnt = 0
    #while fit != 0:
        #cnt += 1
        #print('Attempt number ', cnt)
        #sleep(2)
        #solution = genetic_algorithm(sudoku_matrix)
        #fit = solution.fitness

    #print(solution.matrix)
    #print('Solution found in', cnt, ' attempts')
    
    return
    
    
    
main()



