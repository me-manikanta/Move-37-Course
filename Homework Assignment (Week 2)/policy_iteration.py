from time import time
import numpy as np
import gym

def runPolicy(env, policy):
  state = env.reset()
  done = False
  
  totalReward = 0
  while True:
    state, reward, done, _ = env.step(policy[state])
    totalReward += reward
    if done:
        break
  return totalReward

def evaluatePolicy(env, policy, iterations):
  totalRewards = 0
  for i in range(iterations):
    totalRewards += runPolicy(env, policy)
  return totalRewards / iterations

eps = 1e-10

def constructGreedyPolicy(env, values, gamma):
  policy = np.zeros(env.env.nS)
  for s in range(env.env.nS):
    returns = [
        sum(p * (r + gamma * values[ns])
            for p, ns, r, _ in env.env.P[s][a])
        for a in range(env.env.nA)
    ]
    policy[s] = np.argmax(returns)
  
  return policy

def computeStateValues(env, gamma, policy = None, selectBest = False):  
  values = np.zeros(env.env.nS)
  while True:
    nextValues = values.copy()
    for s in range(env.env.nS):
      if policy is not None:
        action = policy[s]
        nextValues[s] = sum(p * (r + gamma * values[ns]) for p, ns, r, _ in env.env.P[s][action])
      else:
        nextValues[s] = max(
            sum(p * (r + gamma * values[ns])
                for p, ns, r, _ in env.env.P[s][a])
            for a in range(env.env.nA)
        )
      
    diff = np.fabs(nextValues - values).sum()
    values = nextValues
    if diff <= eps:
      break
  return values
