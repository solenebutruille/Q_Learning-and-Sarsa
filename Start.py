
# 1. It renders instance for 500 timesteps, perform random actions
import gym
import random
import numpy as np
import matplotlib.pyplot as pyplot
import sys
import time

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
m = 12
n = 12
res = []
res2 = []
algorithm = "Q-Learning"
nb_problem = 50
nb_episode = 1000
max_nb_step = 200

# Display percentage of advancement
def advance_statut(update):
    sys.stdout.write("\r%d%%" % update)

my_time = time.time()

for k in range(1, nb_problem):
    # For plotting metrics
    earned_reward = 0
    earned_reward_mean = []
    number_of_steps = []
    env = gym.make('gym_foo:foo-v0')
    q_table = np.zeros([(m*n-1), 4])
    for i in range(1, nb_episode):
        env.reset()
        nbStep, state, reward, = 0, 0, 0
        earned_reward = 0
        done = False
        for _ in range(1, max_nb_step):
            if random.uniform(0, 1) < epsilon:
                action2 = env.actionSpaceSample() # Explore action space
                if action2 == 'U':
                    action = 0
                elif action2 == 'D':
                    action = 1
                elif action2 == 'L':
                    action = 2
                elif action2 == 'R':
                    action = 3
            else:
                action = np.argmax(q_table[state]) # Exploit learned values
                if action == 0:
                    action2 = 'U'
                elif action == 1:
                    action2 = 'D'
                elif action == 2:
                    action2 = 'L'
                elif action == 3:
                    action2 = 'R'

            next_state, reward, done, info = env.step(action2)
            next_action = np.argmax(q_table[next_state]) #sarsa
            earned_reward += reward
            next_max = np.max(q_table[next_state])

            if(algorithm == "Sarsa") :
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma*q_table[next_state, next_action] - q_table[state, action])
            elif(algorithm == "Q-Learning"):
                q_table[state, action] = q_table[state, action] + alpha * (reward + gamma*next_max - q_table[state, action]) #Q_Leraning
            else :
                print("unvalid algorithm name")
            state = next_state
            nbStep += 1
            if done:
                break

        number_of_steps.append(nbStep)
        earned_reward_mean.append(earned_reward)

    res.append(number_of_steps)
    res2.append(earned_reward_mean)
    advance_statut(float(i*k) / float(nb_problem*nb_episode) * 100.0)


pyplot.plot(range(0, len(res[0])), np.mean(res, axis=0), "b-", linewidth = 2, label = "Epsilon = 0.1")
pyplot.xlabel("Episode")
pyplot.ylabel("Number of steps")
pyplot.title(algorithm)
pyplot.show()

pyplot.plot(range(0, len(res2[0])), np.mean(res2, axis=0), "b-", linewidth = 2, label = "Epsilon = 0.1")
pyplot.xlabel("Episode")
pyplot.ylabel("Average reward")
pyplot.title(algorithm)
pyplot.show()

#display optimal policy

print('\n')
print("Time = ", time.time() - my_time)
for i in range(m*n):
    action = np.argmax(q_table[i-1]) # Exploit learned values
    if i == 79:
        action2 = 'G'
    elif action == 0:
        action2 = 'U'
    elif action == 1:
        action2 = 'D'
    elif action == 2:
        action2 = 'L'
    elif action == 3:
        action2 = 'R'
    print (action2, end= " ")
    if i in (11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 143):
        print('\n')
