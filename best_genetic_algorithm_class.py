import numpy as np
import matplotlib.pyplot as plt


def fitness_glob(state):
    return np.power(state.sum(), 2)


class BestGA:
    def __init__(self, num_gen, max_value_gen,
                 size_population=100, num_offsprings=100,
                 mutation_rate=0.3, crossover_rate=0.9,
                 num_epoch=100):
        self.num_gen = num_gen # Количество генов
        self.max_value_gen = max_value_gen # Максимальное значение гена
        self.size_population = size_population
        self.num_offsprings = num_offsprings
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_epoch = num_epoch
        
        self.num_parents = int(self.size_population / 2)
        self.num_best_parents = int(self.num_parents / 4) # Часть популяции, которая перейдет в следующее поколение
        self.num_best_offsprings = self.size_population - self.num_best_parents
        self.num_shake = int(self.size_population / 4) # Часть популяции, которая будет встряхиваться
        self.initial_population = self.create_initial_population()
        
    
    def create_initial_population(self):
        return np.random.randint(0, self.max_value_gen, 
                                 size=(self.size_population, self.num_gen))
        
        
    def cal_fitness(self, population, get_fitness):
        """
        Функция оценки приспособленности каждой особи в популяции.
        population : популяция
        get_fitness : ВНЕШНЯЯ ФУНКЦИЯ, которая оценивает приспособленность
        """
        fitness = np.empty(population.shape[0])
        for i in range(fitness.shape[0]):
            fitness[i] = get_fitness(population[i])
        return fitness.astype(int)
        
    
    def selection(self, population, fitness):
        fitness = list(fitness)
        parents = np.empty((self.num_parents, population.shape[1]))
        for i in range(self.num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            parents[i, :] = population[max_fitness_idx[0][0], :]
            fitness[max_fitness_idx[0][0]] = -1000000
        return parents
    
    
    def equality_array(self, x, y):
        """
        Функция проверки схожести двух массивов
        """
        return len([1 for x1, y1 in zip(x, y) if x1 == y1]) == x.shape[0]
    
    
    def crossover(self, parents, get_fitness):
        offsprings = parents[[np.random.randint(0, parents.shape[0]), 
                              np.random.randint(0, parents.shape[0])]].copy() # Две случайныые особи
        crossover_point = np.random.randint(1, parents.shape[1])
        i = 0
        max_iteration = 1e4
        while (offsprings.shape[0] < self.num_offsprings) and (i < max_iteration):
            x = np.random.random()
            if x > self.crossover_rate:
                continue
            parent1_index = i % parents.shape[0]
            parent2_index = (i + 100) % parents.shape[0] #np.random.randint(parents.shape[0])
            children_candidate = np.empty((1, parents.shape[1]))
            children_candidate[0, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
            children_candidate[0, crossover_point:] = parents[parent2_index, crossover_point:]
            #print(parents[parent1_index])
            #print(parents[parent2_index])
            #print(children_candidate)
            #print('=='*10)
            i += 1
            # Проверка, что в популяции детей нет дублирующих особей
            flag_integration = True
            for j in range(offsprings.shape[0]):
                if self.equality_array(children_candidate[0], offsprings[j]):
                    flag_integration = False
                    break
            if flag_integration:
                offsprings = np.vstack((offsprings, children_candidate))
        offsprings = offsprings.astype(int)
        best_offsprings = np.empty((self.num_best_offsprings, parents.shape[1])).astype(int)
        fitness = list(self.cal_fitness(offsprings, get_fitness))
        for i in range(self.num_best_offsprings):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            best_offsprings[i, :] = offsprings[max_fitness_idx[0][0], :]
            fitness[max_fitness_idx[0][0]] = -1000000
        best_offsprings = best_offsprings.astype(int)
        return best_offsprings
    
    
    def mutation(self, offsprings):
        mutants = offsprings.copy()
        for i in range(mutants.shape[0]):
            random_value = np.random.random()
            mutants[i,:] = offsprings[i,:]
            if random_value > self.mutation_rate:
                continue
            int_random_value = np.random.randint(0, offsprings.shape[1])
            mutants[i, int_random_value] = np.random.randint(0, self.max_value_gen)
        mutants = mutants.astype(int)
        return mutants 


    def optimize(self, population, get_fitness):
        fitness_history = []
        for i in range(self.num_epoch):
            fitness_score = self.cal_fitness(population, get_fitness)
            fitness_history.append(fitness_score)
            parents = self.selection(population, fitness_score) # Выбор родителей
            #print(parents)
            #print("=="*10)
            offsprings = self.crossover(parents, get_fitness) # Генерация детей
            mutans = self.mutation(offsprings) # Мутация детей
            #print('mutans', mutans)
            population = parents[:self.num_best_parents]
            population = np.vstack((population, mutans))
            # Встряска популяции каждые 25 эпох.
            population[(self.size_population-self.num_shake):] = \
                np.random.randint(0, self.max_value_gen, size=(self.num_shake, self.num_gen))
                      
        #print('Last generation: \n{}\n'.format(population)) 
        fitness_last_gen = self.cal_fitness(population, get_fitness)      
        #print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
        max_fitness_idx = np.where(fitness_last_gen == np.max(fitness_last_gen))
        best_person = population[max_fitness_idx[0][0]]
        #print('Best person: \n', best_person)
        return best_person, fitness_history
    

if __name__ == "__main__":
    
    BGA = BestGA(25, 2, num_epoch=20)
    best_person, fitness_history = BGA.optimize(BGA.initial_population, fitness_glob)
    
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(BGA.num_epoch)), fitness_history_mean, label = 'Mean Fitness')
    plt.plot(list(range(BGA.num_epoch)), fitness_history_max, label = 'Max Fitness')
    plt.legend(loc='best')
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    






    