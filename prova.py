import numpy as np
from random import randint


class chromosome:
    
    def __init__(self, mat):
        self.matrix = np.copy(mat)
        self.fitness = None

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


def check_grid(matrix, value_list, i, j):
    
    for r in range(i, i+3):
        for c in range(j, j+3):
            if (matrix[r][c] != 0):
                value_list.remove(matrix[r][c])
    
    return

def main():
    mat = np.array([[0,0,0,8,0,0,0,0,9],
                        [0,1,9,0,0,5,8,3,0],
                        [0,4,3,0,1,0,0,0,7],
                        [4,0,0,1,5,0,0,0,3],
                        [0,0,2,7,0,4,0,1,0],
                        [0,8,0,0,9,0,6,0,0],
                        [0,7,0,0,0,6,3,0,0],
                        [0,3,0,0,7,0,0,8,0],
                        [9,0,4,5,0,0,0,0,1]], np.int32)
    
    
    population = list()
    population_dim = 10
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
    
    sub = list()
    sub_dim = len(population) // 2
    if sub_dim % 2:
        sub_dim += 1
        
    for i in range(0, sub_dim):
        best = best_fitness(population)
        sub.append(best)
        population.remove(best)
    
    
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
            
            print('parent1')
            print(parent1.matrix)
            print('parent2')
            print(parent2.matrix)
            print('son')
            print(son.matrix)  

        
    return
    
    
main()