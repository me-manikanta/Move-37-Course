import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.01, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # Q(s, a) <- Q(s, a) + alpha * (reward + lambda * max Q(next_state, a) - Q(s, a))
    def update_q_for_sa(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    #Select an action with epsilon greedy method
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.arg_max(self.q_table[state])
        return action

    #Get the max value from state_actions for collision select some random value
    @staticmethod
    def arg_max(state_actions):
        possibe_actions = []
        max_value = state_actions[0]
        for index, value in enumerate(state_actions):
            if value > max_value:
                possibe_actions.clear()
                max_value = value
                possibe_actions.append(index)
            elif value == max_value:
                possibe_actions.append(index)
        return random.choice(possibe_actions)