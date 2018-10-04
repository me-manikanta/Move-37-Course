import numpy as np
import pprint
from grid_world import GridworldEnv
from policy_evaluation_gridworld import policy_eval

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def lookahead(state, V, discount_factor=1.0):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:

        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):

            chosen_next_action = np.argmax(policy[s])
            action_values = lookahead(s, V, discount_factor)
            best_action = np.argmax(action_values)

            if chosen_next_action != best_action:
                policy_stable = False
            
            policy[s] = np.eye(env.nA)[best_action]

        if policy_stable:
            return policy, V


if __name__ == '__main__':
    policy, v = policy_improvement(env)
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
