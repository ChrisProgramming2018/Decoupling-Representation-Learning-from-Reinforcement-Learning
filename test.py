
import gym

from stable_baselines3 import DQN

env = gym.make("LunarLander-v2")

model = DQN("MlpPolicy", env, verbose=1, batch_size=64, train_freq=1, tau=1e-3, target_update_interval=1, learning_starts=1000)
model.learn(total_timesteps=100000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()
rewards=0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards +=reward
    if done:
        print(rewards)
        break
