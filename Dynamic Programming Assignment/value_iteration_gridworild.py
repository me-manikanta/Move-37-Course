import numpy as np
import pprint
from grid_world import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def lookahead(state, V, discount_factor=1.0):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    

    V = np.zeros(env.nS)

    while True:
        max_v_changed = 0
        for s in range(env.nS):
            A = lookahead(s,V,discount_factor)
            best_action_value = np.max(A)
            max_v_changed = max(max_v_changed, np.abs(best_action_value-V[s]))
            V[s] = best_action_value
    
        if max_v_changed < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    
    for s in range(env.nS):

        A = lookahead(s, V, discount_factor)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    return policy, V

    return policy, V

if __name__ == '__main__':

    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)