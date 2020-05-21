import gym
import itertools
from collections import defaultdict
import numpy as np
import sys
import time
from multiprocessing.pool import ThreadPool as Pool

from collections import defaultdict
import plotting
import matplotlib.pyplot as pyplot


env =  gym.make('gym_foo:foo-v0')
epsilon = 0.1
Q = defaultdict(lambda: np.zeros(env.m*env.n))
E = defaultdict(lambda: np.zeros(env.m*env.n))

def policy_fn(observation):
    nA = env.m*env.n
    A = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[observation])
    A[best_action] += (1.0 - epsilon)
    return A

def sarsa_lambda(env, num_episodes, lamb, discount=0.9, alpha=0.01, epsilon=0.1):

    episode_lengths=np.zeros(num_episodes)
    episode_rewards=np.zeros(num_episodes)
    rewards = [0.]

    for i_episode in range(num_episodes):

        print("\rEpisode {}/{}. ({})".format(i_episode+1, num_episodes, rewards[-1]), end="")
        sys.stdout.flush()

        state = env.reset()
        action_probs = policy_fn(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        while action not in (0,1,2,3):
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():

            if action == 0:
                action2 = 'U'
            elif action == 1:
                action2 = 'D'
            elif action == 2:
                action2 = 'L'
            elif action == 3:
                action2 = 'R'
        #    print("action", action)
            next_state, reward, done, _ = env.step(action2)

            next_action_probs = policy_fn(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            delta = reward + discount*Q[next_state][next_action] - Q[state][action]

            episode_rewards[i_episode] += reward


            E[state][action] += 1

            for s, ac in Q.items():
                E[s][:] *= discount * lamb
                Q[s][:] += alpha * delta * E[s][:]
                if action.all() != ac.all():
                    E[s][:] = 0

            if done:
                break
            else :
                episode_lengths[i_episode] += 1

            state = next_state
            action = next_action

    return Q, episode_lengths, episode_rewards

my_time = time.time()
lamb = 0.9
for _ in range(3):
    rewards = []
    lengths = []
    Q, episode_lengths, episode_rewards = sarsa_lambda(env, 500, lamb)
    rewards.append(episode_rewards)
    lengths.append(episode_lengths)
print('\n')
print("Time = ", time.time() - my_time)
for i in range(12*12):
    next_action_probs = policy_fn(i)
    action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
    if i == 11:
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

pyplot.plot(range(0, len(lengths[0])), np.mean(lengths, axis=0), "b-", linewidth = 2, label = "Epsilon = 0.1")
pyplot.xlabel("Episode")
pyplot.ylabel("Number of steps")
pyplot.title("Sarsa, lambda = 0.9")
pyplot.show()

pyplot.plot(range(0, len(rewards[0])), np.mean(rewards, axis=0), "b-", linewidth = 2, label = "Epsilon = 0.1")
pyplot.xlabel("Episode")
pyplot.ylabel("Average reward")
pyplot.title("Sarsa, lambda = 0.9")
pyplot.show()

next_action_probs = policy_fn(next_state)
