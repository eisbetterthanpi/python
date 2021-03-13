from simple_dqn_tf2 import Agent
import numpy as np
# import gym
from connect4 import connect4
# from utils import plotLearning
import tensorflow as tf
from tensorflow import keras
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_tf2_dqn_lunar_lander.py
# python "F:\main_tf2_dqn_lunar_lander.py"

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    game=connect4()
    lr = 0.001
    n_games = 50#500
    # agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space.shape,
    #     n_actions=env.action_space.n, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=(game.cols,game.rows),
        n_actions=len(game.getValidActions()), mem_size=1000000, batch_size=64, epsilon_end=0.01)
    scores = []
    eps_history = []
    # agent.load_model()
    # agent=keras.model.load_model()
    # agent=agent.load_model()
    for i in range(n_games):
        done = False
        score = 0
        # observation = env.reset()
        # observation=game.board
        observation=game.getValidActions()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

        agent.save_model()

    filename = 'lunarlander_tf2.png'
    x = [i+1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)
