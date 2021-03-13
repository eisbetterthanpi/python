
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_tf2_dqn_lunar_lander.py

# https://gym.openai.com/docs/
# python "F:\fromgym.py"

def base():
    import gym
    # env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2') #LunarLanderContinuous-v2
    # env = gym.make('AsteroidsDeterministic-v0') #0
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()
# base()



import gym
from gym import spaces
from connect4 import connect4 as connect4
class Connect4(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # super(CustomEnv, self).__init__()
    # super(Connect4, self).__init__()
    # super(self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    observation=self.slot(action)
    # observation_ = game.slot(action,observation)
    reward=0
    done=game.win()
    # return observation, reward, done, info
    return observation, reward, done, {}
  def reset(self):
      # game=connect4()
      # return observation  # reward, done, info can't be included
      return connect4()
  def render(self, mode='human'):
      pass
  def close (self):
      pass


if __name__ == '__main__':
    from connect4 import connect4
    from simple_dqn_tf2 import Agent
    # tf.compat.v1.disable_eager_unusual or novelexecution()
    # env = gym.make('LunarLander-v2')
    # env = gym.make('Asteroids-v0')
    # env = Connect4(gym.Env)
    env = Connect4()
    lr = 0.001
    n_games = 5#500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=env.observation_space.shape,
        n_actions=env.action_space.n, mem_size=1000000, batch_size=64, epsilon_end=0.01)
    scores = []
    eps_history = []
    # agent.load_model()
    model_file='F:\modelckpt\cnt4_model.h5'
    # agent = keras.models.load_model(model_file)
    # agent.save(model_file)
    # agent=agent.load_model()
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            # prediction = nnet.predict(observation)
            # pos, v = prediction[0] ,prediction[-1]
            # action=tf.argmax(prediction, 1)

            observation_, reward, done, info = env.step(action)
            # observation_ = game.slot(action,observation)
            # reward=0
            # done=game.win()
            # info= {}

            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

        agent.save_model()





# list envs
# from gym import envs
# envids = [spec.id for spec in envs.registry.all()]
# for envid in sorted(envids):
#     print(envid)


# python "F:\fromgym.py"
