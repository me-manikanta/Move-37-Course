# Just too see how the algo executes faster use smaller constants and execute
# TODO Implement the same in tensorflow

import numpy as np
MAX_IPHONES = 10
MAX_MOVE_OF_IPHONES = 2
IPHONE_PURCHASES_FIRST_LOC = 3
IPHONE_PURCHASES_SECOND_LOC = 2
DELIVERIES_FIRST_LOC = 2
DELIVERIES_SECOND_LOC = 3
DISCOUNT = 0.9
IPHONE_CREDIT = 10
MOVE_IPHONE_COST = 2
actions = np.arange(-MAX_MOVE_OF_IPHONES, MAX_MOVE_OF_IPHONES + 1)
POISSON_UPPER_BOUND = 5

# MAX_IPHONES = 100
# MAX_MOVE_OF_IPHONES = 5
# IPHONE_PURCHASES_FIRST_LOC = 3
# IPHONE_PURCHASES_SECOND_LOC = 4
# DELIVERIES_FIRST_LOC = 3
# DELIVERIES_SECOND_LOC = 2
# DISCOUNT = 0.9
# IPHONE_CREDIT = 10
# MOVE_IPHONE_COST = 2
# actions = np.arange(-MAX_MOVE_OF_IPHONES, MAX_MOVE_OF_IPHONES + 1)
# POISSON_UPPER_BOUND = 11 

poisson_cache = dict()
factorial_cache = dict()

def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = np.exp(-lam) * pow(lam, n) / factorial(n)
    return poisson_cache[key]

def factorial(n):
    if n not in factorial_cache.keys():
        returns = 1
        for i in range(2,n+1):
            returns*=i
        factorial_cache[n] = returns
    return factorial_cache[n]

def expected_return(state, action, state_value, constant_delivered_iphones):
    returns = 0.0
    returns -= MOVE_IPHONE_COST * abs(action)
    for iphone_purchases_first_loc in range(0, POISSON_UPPER_BOUND):
        for iphone_purchases_second_loc in range(0, POISSON_UPPER_BOUND):
            
            num_of_iphones_first_loc = int(min(state[0] - action, MAX_IPHONES))
            num_of_iphones_second_loc = int(min(state[1] + action, MAX_IPHONES))

            real_purchase_first_loc = min(num_of_iphones_first_loc, iphone_purchases_first_loc)
            real_purchase_second_loc = min(num_of_iphones_second_loc, iphone_purchases_second_loc)

            reward = (real_purchase_first_loc + real_purchase_second_loc) * IPHONE_CREDIT

            num_of_iphones_first_loc -= real_purchase_first_loc
            num_of_iphones_second_loc -= real_purchase_second_loc

            prob = poisson(num_of_iphones_first_loc, IPHONE_PURCHASES_FIRST_LOC) * \
                         poisson(num_of_iphones_second_loc, IPHONE_PURCHASES_SECOND_LOC)

            if constant_delivered_iphones:
                delivered_iphones_first_loc = DELIVERIES_FIRST_LOC
                delivered_iphones_second_loc = DELIVERIES_SECOND_LOC
                num_of_iphones_first_loc = min(num_of_iphones_first_loc + delivered_iphones_first_loc, MAX_IPHONES)
                num_of_iphones_second_loc = min(num_of_iphones_second_loc + delivered_iphones_second_loc, MAX_IPHONES)
                returns += prob * (reward + DISCOUNT * state_value[num_of_iphones_first_loc, num_of_iphones_second_loc])
           
    return returns

def policy_iteration(constant_delivered_iphones=True):
    value = np.zeros((MAX_IPHONES + 1, MAX_IPHONES + 1))
    policy = np.zeros(value.shape, dtype=np.int)

    iterations = 1
    while True:
        new_value = np.copy(value)
        for i in range(MAX_IPHONES + 1):
            for j in range(MAX_IPHONES + 1):
                new_value[i, j] = expected_return([i, j], policy[i, j], new_value,
                                                    constant_delivered_iphones)
        value_change = np.abs((new_value - value)).sum()
        print('value change %f' % (value_change))
        value = new_value
        if value_change < 1e-4:
            break

        new_policy = np.copy(policy)
        for i in range(MAX_IPHONES + 1):
            for j in range(MAX_IPHONES + 1):
                action_returns = []
                for action in actions:
                    if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                        action_returns.append(expected_return([i, j], action, value, constant_delivered_iphones))
                    else:
                        action_returns.append(-float('inf'))
                new_policy[i, j] = actions[np.argmax(action_returns)]

        policy_change = (new_policy != policy).sum()
        print('iteration %d : policy changed in %d states' % (iterations, policy_change))
        policy = new_policy
        if policy_change == 0:
            break
        iterations += 1

    print(policy)

if __name__ == '__main__':
    policy_iteration()