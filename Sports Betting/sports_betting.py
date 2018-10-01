import matplotlib.pyplot as plt
import numpy as np
gamma = 1
p = 0.4 # @param
numStates = 100
reward = [0 for _ in range(101)]
reward[100] = 1
theta = 10**-8

value = [0 for _ in range(101)]
policy = [0 for _ in range(101)]

def reinforcement_learning():
    delta = 1
    while delta > theta:
        delta = 0
        for i in range(1, numStates):
            oldValue = value[i]
            bellmanEquation(i)
            diff = abs(oldValue - value[i])
            delta = max(delta, diff)
    print(value)
    plt.figure()
    plt.plot(value[0:100])
    plt.xticks(np.arange(0,101, step=10))
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.title('Value Function')
    plt.show()
    plt.figure()
    plt.plot(policy[0:100])
    plt.xticks(np.arange(0,101, step=10))
    plt.xlabel('Capital')
    plt.ylabel('Final Policy(Stake)')
    plt.title('Final Policy')
    plt.show()


def bellmanEquation(num):
    optimalValue = 0

    for bet in range(0,min(num,100-num)+1):
        win = num+bet
        loss = num-bet
        sum = p * (reward[win] + gamma * value[win]) + (1-p) * (reward[loss] + gamma * value[loss])

        if sum > optimalValue:
            optimalValue = sum
            value[num] = sum
            policy[num] = bet

reinforcement_learning()