#Import blackjack environment
import gym
import sys
env = gym.make("CliffWalking-v0")

import numpy as np
from montecarlo_model import MontaCarloModel

ON_POLICY = 1
OFF_POLICY = 2

#Environment Variables
file_handler = open("cliffwalking_output.txt", "w") 
S = 4 * 12
A = 4
# Running simulation with different conditions
# Method name, method, episodes, epsilon
methods = [
    ('On-Policy Greedy ',ON_POLICY, 100000, 1),
    ('On-Policy epsilon - greedy with eps = 0.3 ', ON_POLICY, 100000, 0.3),
    ('On-Policy epsilon - greedy with eps = 0.6 ', ON_POLICY, 100000, 0.6),
    ('On-Policy epsilon - greedy with eps = 0.9 ', ON_POLICY, 100000, 0.9),
    ('Off-Policy Importance Sampling', OFF_POLICY, 100000, 1)
]

for method_name, method, episodes, epsilon in methods:
    file_handler.writelines(method_name)
    file_handler.writelines('\n')
    m = MontaCarloModel(S, A, epsilon=epsilon)
    if method==ON_POLICY:
        func = m.b
    else:
        func = m.pi
    mod = 100
    for i in range(episodes):
        ep = []
        observation = env.reset() 
        while True:
            action = m.choose_action(func, observation)
            next_observation, reward, done, _ = env.step(action)
            ep.append((observation, action, reward))
            observation = next_observation
            if done:
                break

        m.update_Q(ep)
        m.epsilon = max((episodes-i-1)/episodes, 0.1)

        #The final action would be based always on pi because we are not going to explore 
        if (i+1)%mod ==0 :
            mod *= 10
            file_handler.writelines(format(m.score(env, m.pi, n_samples=10000)))
            file_handler.writelines('\n')

file_handler.close()
