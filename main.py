# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:35:55 2019

@author: marc
"""
# Useful libraries
from print_grid import getGrid, vertical_wall, horizontal_wall
import numpy as np
from copy import deepcopy
from collections import Counter
import random
import matplotlib.pyplot as plt

plan1 = [
        ['Room A'],
        ['Room A', 'A'],
        ['Hall A', 'A'],
        ['Hall B', 'A'],
        ['Room B', 'A'],
        ['Room B', 'A', 'B'],
        ['Hall B', 'A', 'B'],
        ['Hall A', 'A', 'B'],
        ['Room D', 'A', 'B']        
        ]

plan2 = [
        ['Room E', 'F'],
        ['Room E', 'F', 'E'],
        ['Room C', 'F', 'E'],
        ['Room C', 'F', 'E', 'C'],
        ['Hall B', 'F', 'E', 'C'],
        ['Hall A', 'F', 'E', 'C'],
        ['Room D', 'F', 'E', 'C'],
        ['Room D', 'F', 'E', 'C', 'D']        
        ]

plans = [plan1, plan2]

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
QTableNoShaping = [initQTable(len(grid[0]), len(grid)), initQTable(len(grid[0]), len(grid))]

QTableJointPlan = []
Q1 = [initQTable(len(grid[0]), len(grid)) for _ in range(len(plan1))]
Q2 = [initQTable(len(grid[0]), len(grid)) for _ in range(len(plan2))]
QTableJointPlan.append(Q1)
QTableJointPlan.append(Q2)

# Position (x, y) starting with (0, 0) at top left corner
Agent1StartingPosition = np.array([4, 5])
Agent2StartingPosition = np.array([14, 5])
GoalPosition = np.array([1, 11])

# Set parameters
alpha = 0.1
gamma = 0.90
epsilon = 0.1
lambdaelement = 0.4


""" Actions
Action 0 : bottom
Action 1 : left
Action 2 : top
Action 3 : right 
"""

def getAction(QTable, positionAgent):
    p = random.random()
    if( p > epsilon ):
        return bestAction(QTable, positionAgent)
    else:
        return random.randint(0,3)

def bestAction(QTable, position):
    x,y = position
    m = max(QTable[x][y])
    a = [i for i in range(4) if QTable[x][y][i] == m]
    return random.choice(a)

def testIfFinished(positionAgents, listAgents):
    for index in listAgents:
        if positionAgents[index] == GoalPosition:
            del listAgents[index]

def getNewPosition(action, oldPosition):
    """
    From an old position, finds a new position according to a certain action
    """
    newPosition = deepcopy(oldPosition)
    if action == 0:
        newPosition[1] += 1
    elif action == 1:
        newPosition[0] -= 1
    elif action == 2:
        newPosition[1] -= 1
    elif action == 3:
        newPosition[0] += 1
    else:
        exit(1) # We have a problem Houston
    return testIfCorrectPosition(action, oldPosition, newPosition)


def testIfGoesThroughWall(action, oldPosition, newPosition):
    if action == 0:
        walls, _ , _ = grid[oldPosition[1]][oldPosition[0]]
        if horizontal_wall(walls) :
            return True
    elif action == 1:
        walls, _ , _ = grid[oldPosition[1]][oldPosition[0]]
        if vertical_wall(walls) :
            return True
    elif action == 2:
        walls, _ , _ = grid[newPosition[1]][newPosition[0]]
        if horizontal_wall(walls) :
            return True
    elif action == 3:
        walls, _ , _ = grid[newPosition[1]][newPosition[0]]
        if vertical_wall(walls) :
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
    return s[0] == t[0] and s[1] == t[1]

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



def updateQ(QTable, action, reward, currentPosition, newPosition, flags, indexAgent, plan):
    x,y = currentPosition
    i,j = newPosition
    QTable[indexAgent][x][y][action] += alpha * ( reward + gamma * maxQ(QTable, newPosition, indexAgent) -  QTable[indexAgent][x][y][action] )
    
# F(currentPosition, newPosition, flags)
# How do we get currentStepInPlan and TotalStepInPlan ???????????
def F1(currentPosition, newPosition, flags, plan, indexAgent):
    w = 100*len(flags) / len(plan)
    p = [0, 0]
    pos = [newPosition, currentPosition]
    for i in range(2):
        x,y = pos[i]
        for step in plan:
            room, flagsTaken = step[0], step[1:]
            a = [k for k,v in flags.items() if v == indexAgent]  
            if room == grid[y][x][2] and compare(flagsTaken,a):
                p[i] = w * (plan.index(step)+1) 
                break
    p = np.asarray(p)
    coeff = np.asarray([gamma,-1])
    c = coeff * p
    return sum(c)

def maxQ(QTable, newPosition, index):
    x,y = newPosition
    return max(QTable[index][x][y])


"""
Joint-Plan 
"""

def JointPlan(gridFile, episodes = 1000):
    
    rewards = []
    cs = []
    for episode in range(episodes):
        countStep = 0
        flags = {'A': -1, 'B': -1, 'C': -1, 'D': -1, 'E': -1, 'F': -1}
        positionAgents = [Agent1StartingPosition, Agent2StartingPosition]
        listAgents = [0, 1]
        r = 0
        steps = [0, 0]
        while len(listAgents) != 0:
            positionAgents, reward = runOneStepJointPlan(steps, flags, listAgents, positionAgents)
            r += reward
            countStep += 1
        rewards.append(r)
        cs.append(countStep)
    return rewards, cs

def runOneStepJointPlan(step, flags, listAgents, positionAgents):

    newPositionAgents = deepcopy(positionAgents)
    actions, rewards = {}, {}
    
    for indexAgent in listAgents:
        QT = QTableJointPlan[indexAgent][step[indexAgent]]
        actions[indexAgent] = getAction(QT, positionAgents[indexAgent], indexAgent) 
        newPositionAgents[indexAgent] = getNewPosition(actions[indexAgent], positionAgents[indexAgent])
        rewards[indexAgent] = getReward(flags, newPositionAgents[indexAgent], indexAgent) 

    testIfCollision(newPositionAgents, positionAgents)
    updateFlags(newPositionAgents, flags)
    
    for indexAgent in listAgents:
        QT = QTableJointPlan[indexAgent][step[indexAgent]]
        updateQ(QT, actions[indexAgent], rewards[indexAgent], positionAgents[indexAgent], newPositionAgents[indexAgent], flags, indexAgent, plans[indexAgent])
        
        if len(plans[indexAgent][step[indexAgent]]) == 1 and grid[newPositionAgents[indexAgent][1]][newPositionAgents[indexAgent][0]][1] == plans[indexAgent][step[indexAgent]]: # we reach a flag
            step[indexAgent] += 1
        elif len(plans[indexAgent][step[indexAgent]]) > 1 and grid[newPositionAgents[indexAgent][1]][newPositionAgents[indexAgent][0]][2] == plans[indexAgent][step[indexAgent]]: # we reach a room
            step[indexAgent] += 1
            
        if step[indexAgent] == len(plans[indexAgent]):
            listAgents.remove(indexAgent)
            
        
    return newPositionAgents, sum(rewards.values())


"""
No-Shaping plan
"""

def NoShaping(gridFile, episodes = 1000, type = 0):
    
    rewards = []
    cs = []
    t = [[],[]]
    tt = []
    for episode in range(episodes):
        countStep = 0
        flags = {'A': -1, 'B': -1, 'C': -1, 'D': -1, 'E': -1, 'F': -1}
        positionAgents = [Agent1StartingPosition, Agent2StartingPosition]
        listAgents = [0, 1]
        r = 0
        css = [0,0]
        rr = 0
        lam = [[], []]
        while len(listAgents) != 0:
            positionAgents, reward = runOneStepNoShaping(lam, css, flags, listAgents, positionAgents)
            r += sum(reward)
            rr += list(reward)[0]
            countStep += 1
        rewards.append(r)
        t[0].append(css[0])
        t[1].append(css[1])
        cs.append(countStep)
        tt.append(rr)
        print(flags, reward)
        for i in range(2):
            l = lam[i]
            for j in range(len(l)):
                ll = l[j]
                QTableNoShaping[i][ll[0][0]][ll[0][1]][ll[1]] = QTableNoShaping[i][ll[0][0]][ll[0][1]][ll[1]] * (lambdaelement ** j)
    return rewards, cs, t, tt

def runOneStepNoShaping(lam, css, flags, listAgents, positionAgents):

    newPositionAgents = deepcopy(positionAgents)
    actions, rewards = {}, {}
    
    for indexAgent in listAgents:
        css[indexAgent] += 1
        actions[indexAgent] = getAction(QTableNoShaping[indexAgent], positionAgents[indexAgent]) 
        newPositionAgents[indexAgent] = getNewPosition(actions[indexAgent], positionAgents[indexAgent])
        rewards[indexAgent] = getReward(flags, newPositionAgents[indexAgent], indexAgent) 
        lam[indexAgent].insert(0,(positionAgents[indexAgent],actions[indexAgent]))

    testIfCollision(newPositionAgents, positionAgents)
    updateFlags(newPositionAgents, flags)
    for indexAgent in listAgents:
        updateQ(QTableNoShaping, actions[indexAgent], rewards[indexAgent], positionAgents[indexAgent], newPositionAgents[indexAgent], flags, indexAgent, plans[indexAgent])
        if compare(newPositionAgents[indexAgent], GoalPosition):
            listAgents.remove(indexAgent)
    return newPositionAgents, rewards.values()



"""
PRINT RESULTS FUNCTIONS
"""


def getBestMove(pos, index):
    x,y = pos
    action = QTableNoShaping[index][x][y].index(max(QTableNoShaping[index][x][y]))
    p = getNewPosition(action,pos)
    if index == 0:
        print(pos, action, QTableNoShaping[index][x][y], p)
    return p

def printGrid(grid, rewards, countStep):
    m = len(grid)
    n = len(grid[0])
    grid = list(reversed(grid))
    plt.figure()
    plt.xticks(range(1,n+1))
    plt.yticks(range(1,m+1))

    # create a grid
    plt.grid(True, linestyle='-', linewidth=1)
    # put the grid below other plot elements
    seen_rooms = []
    for i in range(m):
        for j in range(n):
            walls, flag, room = grid[i][j]
            if vertical_wall(walls):
                plt.vlines(x=j,ymin=float(i),ymax=float(i+1),linewidth=4, color='k')
            if horizontal_wall(walls):
                plt.hlines(y=i,xmin=float(j),xmax=float(j+1),linewidth=4, color='k')
                
            if flag:
                plt.text(float(j+0.5), float(i+0.5), flag, fontsize=12,horizontalalignment='center', verticalalignment='center')
            if room not in seen_rooms:
                seen_rooms.append(room)
                if room == 'Hall B':
                    plt.text(float(j+0.5), float(i+4.5), room, fontsize=10, weight='bold', verticalalignment='center')
                else:
                    plt.text(float(j+0.5), float(i+0.5), room, fontsize=10, weight='bold', verticalalignment='center')

    plt.text(float(4.15), float(7.5), 'S1', fontsize=10, weight='bold', verticalalignment='center')
    plt.text(float(14.15), float(7.5), 'S2', fontsize=10, weight='bold', verticalalignment='center')
    plt.text(float(0.9), float(1.5), 'Goal', fontsize=8, weight='bold', rotation=-45, verticalalignment='center')

    plt.hlines(y=m, xmin=0 ,xmax=n, linewidth=4,color='k')
    plt.vlines(x=n, ymin=0 ,ymax=m, linewidth=4,color='k')
    
    for j in range(2):
        pos = [ Agent1StartingPosition, Agent2StartingPosition][j]
        current = pos
        x,y = [], []
        i =0
        while not compare(current, GoalPosition) and i < 1000:
            x.append(current[0])
            y.append(current[1])
            current = getBestMove(current, j)
            i +=1
            
        x.append(current[0])
        y.append(current[1])
        for i in range(len(x)):
            x[i] = x[i]+0.5
            y[i] = 12-y[i]+0.5
        plt.plot(x,y)

    plt.show()
    
    # print reward accumulated per episode
    plt.figure(figsize = (10,6))
    plt.xticks(range(0,len(rewards),500))
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    rewards.insert(0,0)
    plt.plot(np.arange(0,len(rewards)), rewards)
    
    #  Print step need to reach goal per episode
    plt.figure( figsize = (10,6))
    plt.xticks(range(0,len(countStep),500))
    plt.xlabel("Episode")
    plt.ylabel("Steps needed to reach goal")
    countStep.insert(0,0)
    plt.plot(np.arange(0,len(countStep)), countStep)
    
    
    plt.show()
    

rewards, cs, t, tt = NoShaping(grid, episodes = 1000)
print("count: ",cs.count(0))
printGrid(grid, rewards, cs)

plt.figure()
plt.plot(np.arange(0,len(t[0])), t[0])
plt.figure()
plt.plot(np.arange(0,len(t[1])), t[1])
plt.figure()
plt.plot(np.arange(0,1000), tt)