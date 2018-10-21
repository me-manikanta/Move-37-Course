from environment import Env
from my_q_learning import QLearningAgent

env = Env()
agent = QLearningAgent(actions = list(range(env.n_actions)))
num_episodes = 1000

for episode in range(num_episodes):

    state = env.reset()

    while True:
        env.render()

        action = agent.get_action(str(state))
        next_state, reward, done = env.step(action)

        agent.update_q_for_sa(str(state), action, reward, str(next_state))

        state = next_state
        env.print_value_all(agent.q_table)

        if done:
            break