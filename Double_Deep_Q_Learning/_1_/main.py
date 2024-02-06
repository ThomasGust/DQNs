from Double_Deep_Q_Learning._1_ import DDQNAgent
import numpy as np
import gym


def main():
    env = gym.make('LunarLander-v2')
    agent = DDQNAgent(alpha=0.0005, gamma=0.99, epsilon=1.0,
                      batch_size=64, input_dims=8, n_actions=env.action_space.n, fname='saved_models/_1_')
    n_games = 1000
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)

            new_observation, reward, done, info = env.step(action=action)
            score += reward

            agent.remember(observation, action, reward, new_observation, done)
            observation = new_observation
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100): i+1])
        print(f'finished episode {i} with a score of {score} and an average score of {avg_score}')

        if i % 10 == 0:
            agent.save_model()
            print(f'successfully saved model at {agent.fname}')


if __name__ == '__main__':
    main()
