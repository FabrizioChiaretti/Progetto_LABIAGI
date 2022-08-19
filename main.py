
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
    dim = 100
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
        prevfit = best_fitness(population)
         
    while (len(population) != 0):
        
        # stopping criterion
        result = best_fitness(population)
        bestfit = result.fitness
        print('best fitness: ', bestfit)
        if result.fitness == 0:
            break
        
        # selection
        sub = list()
        if len(population) % 2 :
            sub_dim = len(population) // 2 +1
        else:
            sub_dim = len(population) // 2
        
        for i in range(0, sub_dim):
            best = best_fitness(population)
            sub.append(best)
    
        # crossover
        l = [i for i in range(0, sub_dim)]
        
        for i in range(0, sub_dim // 2):
            num1 = randint(0, len(l)-1)
            first = l[num1]
            l.remove(first)
            num2 = randint(0, len(l)-1)
            second = l[num2]
            l.remove(second)
            parent1 = sub[first]
            parent2 = sub[second]
            matrix = []
            for i in range (0,5):
                matrix.append(parent1.matrix[i])
            for i in range (5,9):
                matrix.append(parent2.matrix[i])
            
            m = np.array(matrix)
            son = chromosome(m)
            son.fitness = fitness(son)
            population.append(son)
            
        # new popolation generation
            new_dim = sub_dim + sub_dim // 2
            new_population = list()
            for i in range(0, new_dim):
                chrom = best_fitness(population)
                new_population.append(chrom)
                population.remove(chrom)
            population.clear()
            population = new_population
            
        # mutation
        for i in range(0, len(population)):
            num = randint(1, 100)
            chrom = population[i]
            if (num <= 20):
                for j in range(0,9):
                    col = randint(0,8)
                    value = randint(1,9)
                    chrom.matrix[j][col] = value

        prevfit = bestfit
        print(sub_dim, len(population))
    
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
    print(solution)
    
    return
    
    
    
main()



