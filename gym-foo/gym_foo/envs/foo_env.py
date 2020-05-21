import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as pyplot

class FooEnv(gym.Env):
    def __init__(self) :
        m = 12
        n = 12
        self.m = m
        self.n = n
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        self.agentStartStatePossible = [60, 72, 120, 132]
        self.agentPosition = np.random.choice(self.agentStartStatePossible)
        self.finalState = 11 #case A : 11, case B : 33, case C : 79
        self.zone1 = [28, 29, 30, 31, 32, 33, 40, 52, 64, 76, 88, 100, 101, 102, 103, 104, 45, 57, 69, 81, 80, 92]
        self.zone2 = [41, 42, 43, 44, 53, 65, 77, 89, 90, 91, 79, 67, 68, 56]
        self.zone3 = [55, 54, 66, 78]


    def isTerminalState(self, state):
        return self.agentPosition == self.finalState

    def offGridMove(self, oldState, action):
        resultingState = self.agentPosition + self.actionSpace[action]
        if resultingState not in range(0, (self.m*self.n-1)):
            return True
        elif (oldState+1) % 12 == 0 and action == 'R':
            return True
        elif oldState in range(0, 11) and action == 'U':
            return True
        elif oldState in range(132, (self.m*self.n-1)) and action == 'D':
            return True
        elif (oldState) % 12 == 0 and action == 'L':
            return True
        else:
            return False

    def getReward(self, resultingState) :
        reward = 0
        if self.isTerminalState(resultingState) :
            reward = 10
        elif resultingState in self.zone1 :
            reward = -1
        elif resultingState in self.zone2 :
            reward = -2
        elif resultingState in self.zone3 :
            reward = -3
        return reward

    def grid_parameters(self, action):
        epsilon = np.random.random()
        if epsilon < 0.1:
            actions = ['U', 'D', 'L', 'R']
            actions.remove(action)
            action = np.random.choice(actions)
        if epsilon > 0.5 and not self.offGridMove(self.agentPosition, 'R'):
            self.agentPosition += 1
        return action

    def step(self, action):
        action = self.grid_parameters(action)
        resultingState = self.agentPosition + self.actionSpace[action]
        if not self.offGridMove(self.agentPosition, action):
            self.agentPosition = resultingState
            reward = self.getReward(resultingState)
            return resultingState, reward, self.isTerminalState(self.agentPosition), None
        else :
            reward = self.getReward(self.agentPosition)
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None


    def reset(self):
        self.agentPosition = np.random.choice(self.agentStartStatePossible)
        self.grid = np.zeros((self.m, self.n))
        return self.agentPosition

    def render(self):
        print('-------------------------')
        for row in range(self.m*self.n):
                if row == self.agentPosition:
                    print('X', end= '   ')
                elif row in self.zone1 :
                    print('1', end= '   ')
                elif row in self.zone2 :
                    print('2', end= '   ')
                elif row in self.zone3 :
                    print('3', end= '   ')
                elif row == self.finalState :
                    print('G', end= '   ')
                else:
                    print('-', end = '   ')
                if (row+1) % 12 == 0:
                    print('\n')
        print('\n-------------------------')


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)
