
import numpy as np
from random import randint


class chromosome:
    
    def __init__(self, mat):
        self.matrix = np.copy(mat)
        self.fitness = None
        

def best_fitness(population):
    result = None
    best_fitness = 1000
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


def genetic_algorithm(mat):
    
    result = None
    
    # initialize population
    population = list()
    dim = 32
    for i in range(0,dim):
        
        chrom = chromosome(mat)
        for i in range(0,9):
            for j in range(0,9):
                if chrom.matrix[i][j] != 0:
                    continue
                else:
                    chrom.matrix[i][j] = randint(1,9)
        
        chrom.fitness = fitness(chrom)
        population.append(chrom)  
         
    while (1):
        
        # stopping criterion
        result = best_fitness(population)
        if result.fitness == 0:
            break
        
        # selection
        new_dim = len(population) // 2
        if new_dim % 2 != 0:
            new_dim += 1
            print(new_dim)
    
        new_population = list()
        for i in range(0, new_dim):
            best = best_fitness(population)
            population.remove(best)
            new_population.append(best)
    
        population.clear()
        population = new_population
    
        # crossover
        
    
    
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
                        [0,0,3,2,0,0,0,0,0],
                        [0,0,0,0,0,1,0,0,0]], np.int32)
    
    
    solution = genetic_algorithm(matrix1)
    #print(solution)
    
    return
    
    
    
main()



