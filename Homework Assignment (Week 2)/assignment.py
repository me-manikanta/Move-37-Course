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


def valueIteration(env, gamma):
  stateValues = computeStateValues(env, gamma, selectBest=True)
  policy = constructGreedyPolicy(env, stateValues, gamma)
  return policy

def randomPolicy(env):
  return np.random.choice(env.env.nA, size=(env.env.nS))
  
def policyIteration(env, gamma):
  policy = randomPolicy(env)
  while True:
    stateValues = computeStateValues(env, gamma, policy)
    nextPolicy = constructGreedyPolicy(env, stateValues, gamma)
    if np.all(policy == nextPolicy):
      break
    policy = nextPolicy
      
  return policy

evaluateIterations = 1000

def solveEnv(env, methods, envName):
  print(f'Solving environment {envName}')
  for method in methods:
    name, f, gamma = method
    tstart = time()
    policy = f(env, gamma)
    tend = time()
    print(f'It took {tend - tstart} seconds to compute a policy using "{name}" with gamma={gamma}')
    
    score = evaluatePolicy(env, policy, evaluateIterations)
    print(f'Policy average reward is {score}')
  
  
methods = [
  ('Value Iteration', valueIteration, 0.9),
  ('Policy Iteration', policyIteration, 0.9),
  ('Value Iteration', valueIteration, 0.98),
  ('Policy Iteration', policyIteration, 0.98),
  ('Value Iteration', valueIteration, 1),
  ('Policy Iteration', policyIteration, 1),
]

env = gym.make('FrozenLake-v0')
solveEnv(env, methods, 'Frozen Lake 4x4')