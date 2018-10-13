from collections import defaultdict
from copy import deepcopy
import numpy as np

ON_POLICY = 1
OFF_POLICY = 2

class MontaCarloModel:
    def __init__(self, state_space, action_space, gamma=1.0, epsilon =0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.C = defaultdict(lambda: np.zeros(action_space.n))
        if isinstance(action_space, int):
            self.action_space = np.arange(action_space)
            actions = [0]*action_space
            self._act_rep = "list"
        else:
            self.action_space = action_space
            actions = {k:0 for k in action_space}
            self._act_rep = "dict"
        if isinstance(state_space, int):
            self.state_space = np.arange(state_space)
            self.Q = [deepcopy(actions) for _ in range(state_space)]
        else:
            self.state_space = state_space
            self.Q = {k: deepcopy(actions) for k in state_space}
            
        self.Ql = deepcopy(self.Q)

    #Action probability determined randomly
    def random_action(self, action, state):
        return 1

    #Action probability determined by Bellman Equation
    def pi(self, action, state):
        
        if self._act_rep == "list":
            if action == np.argmax(self.Q[state]):
                return 1
            return 0
        elif self._act_rep == "dict":
            if action == max(self.Q[state], key=self.Q[state].get):
                return 1
            return 0

    #Action probabilty determined by ϵ - Greedy Policy(On - Policy) 
    def b(self, action, state):
        return self.epsilon/len(self.action_space) + (1-self.epsilon) * self.pi(action, state)

    def action_probabilitites(self, policy, state):
        probability_of_actions = [policy(action, state) for action in self.action_space]
        return probability_of_actions

    #Depending upon what policy you pick it returns the action
    def choose_action(self, policy, state):

        probability_of_actions = self.action_probabilitites(policy, state)
        return np.random.choice(self.action_space, p=probability_of_actions)

    #Inititalise returns for each episode
    def generate_returns(self, ep):
        G = {}
        C = 0
        for observation, action, reward in ep:
            C = reward + self.gamma * C
            G[(observation, action)] = C
        return G

    #update Q after every episode
    def update_Q(self, ep, method=ON_POLICY):
        if method == ON_POLICY:
            G = self.generate_returns(ep)
            for state, action in G:
                q = self.Q[state][action]
                self.Ql[state][action] += 1
                N = self.Ql[state][action]
                self.Q[state][action] = q * N/(N+1) + G[(state, action)]/(N+1)
        else:
            for t in range(len(ep))[::-1]:
                state, action, reward = ep[t]
                G = 0.1 * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                if action !=  np.argmax(self.action_probabilitites(self.pi, state)):
                    break
                W = W * 1.0/self.action_probabilitites(self.pi, state)[action]

    #Evaluate the score of policy
    def score(self, env, policy, n_samples=10000):
        rewards = []
        rewards = []
        for _ in range(n_samples):
            observation = env.reset()
            cum_rewards = 0
            while True:
                action = self.choose_action(policy, observation)
                observation, reward, done, _ = env.step(action)
                cum_rewards += reward
                if done:
                    rewards.append(cum_rewards)
                    break
        return np.mean(rewards)
