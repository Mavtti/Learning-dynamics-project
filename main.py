from planBasedReward.print_grid import *
import numpy as np
from copy import deepcopy

grid = getGrid("planBasedReward/grid.txt")

Q = [] # Create Q table
for j in range(13):
    Q.append([])
    for i in range(18):
        Q[j].append([0] * 4)

# Position (x, y) starting with (0, 0) at top left corner
Agent1StartingPosition = np.array([4, 5])
Agent2StartingPosition = np.array([14, 5])
GoalPosition = np.array([1, 11])

alpha = 0.1
gamma = 0.99
epsilon = 0.1
lambdaelement = 0.4


# Actions
# Action 0 : bottom
# Action 1 : left
# Action 2 : top
# Action 3 : right 

def getAction(positionAgent):
    return 1

def testIfFinished(positionAgents, listAgents):
    for index in listAgents:
        if positionAgents[index] == GoalPosition:
            del listAgents[index]

def getNewPosition(action, oldPosition):
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
        walls, _ , _ = grid[oldPosition[0]][oldPosition[1]]
        if horizontal_wall(walls):
            return True
    elif action == 1:
        walls, _ , _ = grid[oldPosition[0]][oldPosition[1]]
        if vertical_wall(walls):
            return True
    elif action == 2:
        walls, _ , _ = grid[newPosition[0]][newPosition[1]]
        if horizontal_wall(walls):
            return True
    elif action == 3:
        walls, _ , _ = grid[newPosition[0]][newPosition[1]]
        if vertical_wall(walls):
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
            if newPositionAgents[j] == newPositionAgents[i]:
                collision.append(i)
                collision.append(j)
        for agent in collision:
            newPositionAgents[agent] = deepcopy(oldPositionAgents[agent])

def getReward(flags, newPosition, indexAgent):
    if newPosition == GoalPosition:
        return flags.count(indexAgent) * 100
    else:
        return 0

def updateFlags(newPositionAgents, flags):
    for i in range(len(newPositionAgents)):
        positionAgent = newPositionAgents[i]
        _, flag, _ = grid[positionAgent[0]][positionAgent[1]]
        if flag and flags.get(flag) != -1:
            flags[flag] = i

def updateQ():
    return

def runOneStep(flags, listAgents, positionAgents, currentStep):
    # actions = []
    newPositionAgents = deepcopy(positionAgents)
    for indexAgent in listAgents:
        action = getAction(positionAgents[indexAgent])
        newPositionAgents[indexAgent] = getNewPosition(action, positionAgents[indexAgent])
        getReward(flags, newPositionAgents[indexAgent], indexAgent)
        updateQ()
        # actions.append(action)
    testIfCollision(newPositionAgents, positionAgents)
    updateFlags(newPositionAgents, flags)
    return positionAgents



def planBasedRewardLearning(episodes = 100):
    for episode in range(episodes):
        currentStep = 0
        flags = {'A': -1, 'B': -1, 'C': -1, 'D': -1, 'E': -1, 'F': -1}
        positionAgents = [Agent1StartingPosition, Agent2StartingPosition]
        listAgents = [0, 1]
        while len(listAgents) != 0:
            # On fais une étape
            positionAgents = runOneStep(flags, listAgents, positionAgents, currentStep)
            currentStep += 1

    return True

