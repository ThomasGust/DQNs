from Simple_DQN._1_ import SimpleDQNAgent
import numpy as np
import gym


def train_dqn():
    env = gym.make('LunarLander-v2')
    n_games = 500
    agent = SimpleDQNAgent(discount_factor=0.99, epsilon=1.0, learning_rate=0.0005,
                           input_dims=8, n_actions=env.action_space.n, mem_size=1000000,
                           batch_size=64, epsilon_end=0.01)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0

        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.remember(state=observation, new_state=observation_,
                           action=action, reward=reward, done=done)
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):i+1])
        print(f'episode {i}, score {score}, average score {avg_score}')

def main():
    train_dqn()


if __name__ == '__main__':
    main()
