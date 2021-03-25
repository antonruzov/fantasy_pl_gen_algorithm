import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import best_genetic_algorithm_class as ga


class BestFantasyTeam:
    def __init__(self, points, cost, position, 
                 max_cost_team=850, 
                 sheme={'GK' : 1, 'DEF' : 4, 'MID' : 4, 'FWD' : 2}):
        self.points = points # Array points
        self.cost = cost # Array cost
        self.position = position # Array position
        self.max_cost_team = max_cost_team # Float : max cost team
        self.sheme = sheme
        
    
    def get_count_error_position(self, state):
        """
        Функция оценки занятых позиций в составе.
        state :: массив, индексы игроков, попадающих в состав.
        return :: count_error
        """
        fitness_position = 0
        current_position = list(self.position[state])
        for pos, count_players in self.sheme.items():
            if count_players != current_position.count(pos):
                fitness_position = -10000
                break
        return fitness_position
    
    
    def get_score_team_cost(self, state):
        """
        Функция оценки стоимости состава.
        state :: массив, индексы игроков, попадающих в состав.
        return :: score
        """
        current_cost = self.cost[state]
        if current_cost.sum() > self.max_cost_team:
            #return - (current_cost.sum() - self.max_cost_team) ** 2
            return -10000
        else:
            return 0
        
    
    def get_double_players(self, state):
        """
        Функция, исключающая возможность повторения игроков.
        state :: массив, индексы игроков, попадающих в состав.
        return :: count_error
        """
        if len(set(state)) != len(state):
            #return len(state) - len(set(state))
            return -10000
        else:
            return 0
        
        
    def total_points(self, state):
        """
        Функция оценки очков состава.
        state :: массив, индексы игроков, попадающих в состав.
        """
        return self.points[state].sum()
        
    
    def fitness_glob(self, state):
        """
        Функция оценки состояния
        return :: score fitness_glob
        """
        base_fitness = 1
        state = state.astype(int)
        fitness_glob = self.get_count_error_position(state) * base_fitness + \
                       self.get_score_team_cost(state) + \
                       self.get_double_players(state) * base_fitness + \
                       self.total_points(state)
        return fitness_glob
    

def plotting_learning_epoch(fitness_history, num_epoch=100):
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(num_epoch)), fitness_history_mean, label = 'Mean Fitness')
    plt.plot(list(range(num_epoch)), fitness_history_max, label = 'Max Fitness')
    plt.legend(loc='best')
    plt.title('Fitness through the generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    
    

if __name__ == "__main__":
    
    cleaned_players = pd.read_csv('cleaned_players.csv')
    cleaned_players['first_second_name'] = cleaned_players['first_name'] + cleaned_players['second_name']
    points = cleaned_players['total_points'].values
    cost = cleaned_players['now_cost'].values
    position = cleaned_players['element_type'].values
    
    BST = BestFantasyTeam(points, cost, position)
    
    count_playes_squad = 11
    max_idx = cleaned_players.shape[0]
    num_iteration = 2
    cost_team = []
    players = []
    iteration = []
    score_team = []
    
    for i in range(num_iteration):
        print('number_iteration', i)
        BGA = ga.BestGA(num_gen=count_playes_squad, max_value_gen=max_idx, num_epoch=200, size_population=500)
        best_person, fitness_history = BGA.optimize(BGA.initial_population, BST.fitness_glob)
        
        #plotting_learning_epoch(fitness_history, BGA.num_epoch)
        
        best_person = best_person.astype(int)
        cost_team.extend([BST.cost[best_person].sum() for k in range(count_playes_squad)])
        players.extend([player[0] for player in cleaned_players[['first_second_name']].values[best_person]])
        score_team.extend([BST.points[best_person].sum() for k in range(count_playes_squad)])
        iteration.extend([i for k in range(count_playes_squad)])
    
    result_df = pd.DataFrame(np.column_stack((iteration, players, cost_team, score_team)),
                             columns=['iter', 'player', 'cost_team', 'score_team'])
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    