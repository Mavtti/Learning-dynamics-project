# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:35:55 2019

@author: marc
"""
# Useful libraries
from print_grid import getGrid, printGrid
import numpy as np
from copy import deepcopy
from collections import Counter
import random

# Load the grid
grid = getGrid("grid.txt")

def initQTable(n,m):
    QTable = []
    for i in range(n):
        QTable.append([])
        for _ in range(m):
            QTable[i].append([0, 0, 0, 0])
    return QTable

# Initialize Q-Table
QTable = initQTable(len(grid[0]), len(grid))

# Position (x, y) starting with (0, 0) at top left corner
Agent1StartingPosition = np.array([4, 7])
Agent2StartingPosition = np.array([14, 7])
GoalPosition = np.array([1, 11])

# Set parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
lambdaelement = 0.4


""" Actions
Action 0 : bottom
Action 1 : left
Action 2 : top
Action 3 : right 
"""

def getAction(positionAgent):
    p = random.random()
    if( p > epsilon ):
        return bestAction(positionAgent)
    else:
        return random.randint(0,3)

def bestAction(position):
    x,y = position
    m = max(QTable[x][y])
    return QTable[x][y].index(m)

def testIfFinished(positionAgents, listAgents):
    for index in listAgents:
        if positionAgents[index] == GoalPosition:
            del listAgents[index]

def getNewPosition(action, oldPosition):
    newPosition = deepcopy(oldPosition)
    if action == 0:
        newPosition[1] -= 1
    elif action == 1:
        newPosition[0] -= 1
    elif action == 2:
        newPosition[1] += 1
    elif action == 3:
        newPosition[0] += 1
    else:
        exit(1) # We have a problem Houston
    return testIfCorrectPosition(action, oldPosition, newPosition)

def testIfGoesThroughWall(action, oldPosition, newPosition):
    if action == 0:
        walls, _ , _ = grid[oldPosition[1]][oldPosition[0]]
        if walls in [2,3] :
            return True
    elif action == 1:
        walls, _ , _ = grid[oldPosition[1]][oldPosition[0]]
        if walls in [1,3]:
            return True
        walls, _ , _ = grid[newPosition[1]][newPosition[0]]
        if walls in [1,3]:
            return True
    elif action == 3:
        walls, _ , _ = grid[newPosition[1]][newPosition[0]]
        if walls in [2,3]:
            return True
    return False

def testIfCorrectPosition(action, oldPosition, newPosition):
    if newPosition[0] < 0:
        return [0, newPosition[1]]
    elif newPosition[0] > 17:
        return [17, newPosition[1]]
    elif newPosition[1] < 0:
        return [newPosition[0], 0]
    elif newPosition[1] > 12:
        return [newPosition[0], 12]
    elif testIfGoesThroughWall(action, oldPosition, newPosition):
        return deepcopy(oldPosition)
    return newPosition

def testIfCollision(newPositionAgents, oldPositionAgents):
    for i in range(len(newPositionAgents)):
        collision = []
        for j in range(i+1, len(newPositionAgents)):
            b1 = compare(newPositionAgents[j], newPositionAgents[i]) # check if agents go to the same positon
            b2 = compare(oldPositionAgents[j], newPositionAgents[i]) and compare(newPositionAgents[j], oldPositionAgents[i]) #check if agents swap position
            if b1 or b2:
                collision.append(i)
                collision.append(j)
    collision = list(set(collision)) # get unique values in the list
    for agent in collision:
        newPositionAgents[agent] = deepcopy(oldPositionAgents[agent])

def compare(s, t):
    return Counter(s) == Counter(t)

def getReward(flags, newPosition, indexAgent):
    if compare(newPosition, GoalPosition):
        return Counter(flags.values())[indexAgent] * 100
    else:
        return 0

def updateFlags(newPositionAgents, flags):
    for i in range(len(newPositionAgents)):
        positionAgent = newPositionAgents[i]
        _, flag, _ = grid[positionAgent[1]][positionAgent[0]]
        if flag and flags.get(flag) == -1:
            flags[flag] = i


def runOneStep(flags, listAgents, positionAgents, currentStep):

    newPositionAgents = deepcopy(positionAgents)
    for indexAgent in listAgents:
        action = getAction(positionAgents[indexAgent]) 
        newPositionAgents[indexAgent] = getNewPosition(action, positionAgents[indexAgent])
        reward = getReward(flags, newPositionAgents[indexAgent], indexAgent)
        updateQ(action, reward, positionAgents[indexAgent], newPositionAgents[indexAgent], flags, indexAgent)
        # actions.append(action)
        if reward != 0:
            listAgents.remove(indexAgent) # if an agent reaches the goal 
    testIfCollision(newPositionAgents, positionAgents)
    updateFlags(newPositionAgents, flags)
    return newPositionAgents

def updateQ(action, reward, currentPosition, newPosition, flags, indexAgent):
    x,y = currentPosition
    i,j = newPosition
    QTable[x][y][action] += alpha * ( reward - 0.5 + gamma * maxQ(newPosition, flags, indexAgent) -  QTable[x][y][action] )
    
# F(currentPosition, newPosition, flags)
# How do we get currentStepInPlan and TotalStepInPlan ???????????
def F(currentPosition, newPosition):
    w = 600 
    np = w * 0
    p = 0
    return gamma * np - p

def maxQ(newPosition, flags, indexAgent):
    rewards= []
    for i in range(4):
        position = getNewPosition(i, newPosition)
        rewards.append(getReward(flags, position, indexAgent))
    return max(rewards)

def planBasedRewardLearning(gridFile, episodes = 100):
    
    for episode in range(episodes):
        currentStep = 0
        flags = {'A': -1, 'B': -1, 'C': -1, 'D': -1, 'E': -1, 'F': -1}
        positionAgents = [Agent1StartingPosition, Agent2StartingPosition]
        listAgents = [0, 1]
        while len(listAgents) != 0:
            positionAgents = runOneStep(flags, listAgents, positionAgents, currentStep)
            currentStep += 1

    return True

def getBestMove(pos):
    x,y = pos
    action = QTable[x][y].index(max(QTable[x][y]))
    return getNewPosition(action,pos)


def printOnMyWay():
    plt = printGrid(grid)
    for pos in [ Agent1StartingPosition, Agent2StartingPosition]:
        current = pos
        l = []
        print("########################") # BOUCLE INFINIE !!!!!!!!!!!!!!!!
        while not compare(current, GoalPosition):
            l.append(current)
            print(current)
            current = getBestMove(current)
        l.append(current)
        l = np.asarray(l)
        plt.plot(l[:,0], l[:,1])
    plt.show()
    