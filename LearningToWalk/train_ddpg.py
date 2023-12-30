import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def main():
    run_name = 'ddpg_walker_1m'

    env = gym.make('Walker2d-v4', render_mode='rgb_array')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log='logs/{}'.format(run_name))
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save('models/{}'.format(run_name))

    render_env = HumanRendering(model.get_env())
    obs = render_env.reset()

    while True:
        action, _states = model.predict(obs)  
        obs, rewards, done, info = render_env.step(action)
        render_env.render()

    render_env.close()

if __name__ == "__main__":
  	main()