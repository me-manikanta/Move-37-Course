import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid
from utils import max_dict, print_values, print_policy

GAMMA = 0.9
EPSILON=0.20
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
N_EPISODES = 10000

def epsilon_action(a, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
    s = (2,0)
    grid.set_state(s)
    a = epsilon_action(policy[s], EPSILON)

    states_actions_rewards = [(s, a, 0)]
    while True:
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = epsilon_action(policy[s], EPSILON)
            states_actions_rewards.append((s,a,r))
    G = 0
    states_actions_returns = []
    first = True
    for s,a,r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r+GAMMA*G
    states_actions_returns.reverse()
    return states_actions_returns

def monte_carlo(grid):
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    Q = {}
    returns = {}
    states = grid.non_terminal_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            returns[(s,a)] = []
    
    deltas = []
    for t in range(N_EPISODES):
        if t%1000 == 0:
            print(t)

        biggest_change = 0
        states_actions_returns = play_game(grid, policy)

        seen_state_action_pairs = set()

        for s, a, G in states_actions_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                returns[sa].append(G)
                old_q = Q[s][a]
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        for s in policy.keys():
            a, _ = max_dict(Q[s])
            policy[s]=a
    V = {}
    for s in policy.keys():
        V[s] = max_dict(Q[s])[1]
    
    return V, policy, deltas

if __name__ == '__main__':
    grid = standard_grid(obey_prob=1.0, step_cost=None)

    print("rewards:")
    print_values(grid.rewards, grid)

    V, policy, deltas = monte_carlo(grid)

    print("final values:")
    print_values(V, grid)
    print("final policy:")
    print_policy(policy, grid)

    plt.plot(deltas)
    plt.show()
