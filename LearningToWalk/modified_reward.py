import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import numpy as np
import time
from typing import Callable

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def main():
    run_name = 'walker_lr_scheduler_x0.1_1m'

    env = gym.make('Walker2d-v4', render_mode='rgb_array')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    start_time = time.time()
    print('Start time:', start_time)
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=linear_schedule(0.0001), tensorboard_log='logs/{}'.format(run_name))
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save('models/{}'.format(run_name))
    
    end_time = time.time()
    print('End time:', end_time)
    elapsed_time = end_time - start_time
    print('Total runtime:', elapsed_time)

    render_env = HumanRendering(model.get_env())
    obs = render_env.reset()

    while True:
        action, _states = model.predict(obs)  
        obs, rewards, done, info = render_env.step(action)
        render_env.render()

    render_env.close()

if __name__ == "__main__":
  	main()