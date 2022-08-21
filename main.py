
import numpy as np
from random import randint
from time import sleep


class chromosome:
    
    def __init__(self, mat):
        self.matrix = np.copy(mat)
        self.fitness = None
        

def best_fitness(population):
    result = None
    best_fitness = 800
    n = len(population)
    
    for i in range(0, n):
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
    
    result = None
    
    # initialize population
    population = list()
    population_dim = 800
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
    
    c = best_fitness(population)
    prevfit = c.fitness
         
    while (len(population) != 0):
        
        # stopping criterion
        result = best_fitness(population)
        bestfit = result.fitness
        print('population_dim: ', len(population))
        print('update: ', prevfit - bestfit)
        print('best fitness: ', bestfit)
        if result.fitness == 0:
            print('Solution found !!')
            break
        
        # selection
        sub = list()
        sub_dim = len(population) // 2
        if sub_dim % 2:
            sub_dim += 1
        
        for i in range(0, sub_dim):
            best = best_fitness(population)
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
            population.append(sub[num2])
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
            population.append(son)
            
        population_dim = len(population)
        
        #if len(population) != (population_dim + sub_dim // 2):
            #print('problema')
            #sleep(10)
         
        # new popolation generation
        best = best_fitness(population)
        population_dim -= (sub_dim // 2)
        if best.fitness < bestfit:
            population_dim -= 1
        new_population = list()
        for i in range(0, population_dim):
            chrom = best_fitness(population)
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
                
        prevfit = bestfit
        #sleep(1)
    
    print('genetic algorithm finished')
    
    return result



def main ():
    
    matrix1 = np.array([[0,0,0,8,0,0,0,0,9],
                        [0,1,9,0,0,5,8,3,0],
                        [0,4,3,0,1,0,0,0,7],
                        [4,0,0,1,5,0,0,0,3],
                        [0,0,2,7,0,4,0,1,0],
                        [0,8,0,0,9,0,6,0,0],
                        [0,7,0,0,0,6,3,0,0],
                        [0,3,0,0,7,0,0,8,0],
                        [9,0,4,5,0,0,0,0,1]], np.int32)
    
    
    matrix2 = np.array([[2,0,5,0,0,9,0,0,4],
                        [0,0,0,0,0,0,3,0,7],
                        [7,0,0,8,5,6,0,1,0],
                        [4,5,0,7,0,0,0,0,0],
                        [0,0,9,0,0,0,1,0,0],
                        [0,0,0,0,0,2,0,8,5],
                        [0,2,0,4,1,8,0,0,6],
                        [6,0,8,0,0,0,0,0,0],
                        [1,0,0,2,0,0,7,0,8]], np.int32)
    
    
    matrix3 = np.array([[0,0,6,0,9,0,2,0,0],
                        [0,0,0,7,0,2,0,0,0],
                        [0,9,0,5,0,8,0,7,0],
                        [9,0,0,0,3,0,0,0,6],
                        [7,5,0,0,0,0,0,1,9],
                        [1,0,0,0,4,0,0,0,5],
                        [0,1,0,3,0,9,0,8,0],
                        [0,0,0,2,0,1,0,0,0],
                        [0,0,9,0,8,0,1,0,0]], np.int32)
    
    
    matrix4 = np.array([[0,0,0,8,0,0,0,0,0],
                        [7,8,9,0,1,0,0,0,6],
                        [0,0,0,0,0,6,1,0,0],
                        [0,0,7,0,0,0,0,5,0],
                        [5,0,8,7,0,9,3,0,4],
                        [0,4,0,0,0,0,2,0,0],
                        [0,0,3,2,0,0,0,0,0],
                        [8,0,0,0,7,0,4,3,9],
                        [0,0,0,0,0,1,0,0,0]], np.int32)
    
    
    solution = genetic_algorithm(matrix3)
    print(solution.matrix)
    
    return
    
    
    
main()



