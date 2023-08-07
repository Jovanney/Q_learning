import numpy as np
import random
from connection import connect, get_state_reward
import os

# Define os hiperparâmetros
alpha = 0.5 # taxa de aprendizado 
gamma = 0.95 # taxa de desconto
epsilon = 0.3 # taxa de exploração

min_epsilon = 0.2
epsilon_decay = 0.005

# Inicializando a tabela Q (96 estados x 3 ações)
# 96 estados: 24 plataformas x 4 direções. 3 ações: esquerda, direita e pular
def load_matrix():
    with open(f'{os.getcwd()}/resultado.txt') as file:
        matrix = [[float(n) for n in line.split()] for line in file]
    return matrix

# Salva a matriz
def save_matrix(Q_table):
    with open(f'{os.getcwd()}/resultado.txt', 'w') as file:
        txt = ''.join([f'{round(state[0], 6)} {round(state[1], 6)} {round(state[2], 6)}\n' for state in Q_table])
        file.write(txt)

# Constrói a tabela Q 
Q_table = load_matrix()

victories = 0

reward_per_episode = []

# Conecta-se ao servidor local do jogo
s = connect(2037) 

# Número de episódios = 100000
for episode in range (1,100001):
    print(f'episódio: {episode} vitórias: {victories}')
    state = 0b0000000 # Estado inicial
    done = False
    total_reward = 0

    while not done:
        action = random.randint(0,2) if random.uniform(0, 1) < epsilon else np.argmax(Q_table[state]) # Seleciona uma ação aleatória ou explora valores aprendidos
        
        act = ["left", "right", "jump"][action]

        old_value = Q_table[state][action]
        # Obtém o próximo estado e recompensa
        next_state, reward = get_state_reward(s , act)

        next_state = int(next_state, 2) # Converte para inteiro
        next_max = np.max(Q_table[next_state])

        # Equação de Otimização de Bellman
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q_table[state][action] = new_value
        total_reward += reward

        # Atualiza o estado atual
        state = next_state
        save_matrix(Q_table)

        # É feito se o agente cair ou atingir o objetivo
        if reward == -100 or reward == 300:
            done = True

    # Atualiza o decaimento
    epsilon -= epsilon_decay
    
    # Recompensa total quando os agentes atingem o objetivo e contam o número de vitórias
    if reward == 300:
        total_reward = 700 - total_reward
        victories += 1

    if epsilon <= min_epsilon:
        epsilon = 0.4

    # Armazena a recompensa total toda vez que o agente cai ou atinge o objetivo 
    reward_per_episode.append(total_reward)
