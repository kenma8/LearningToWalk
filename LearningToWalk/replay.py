import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import numpy as np
import time
from typing import Callable
from statistics import mean

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    run_name = 'walker_lr_scheduler_x0.1_1m'

    env = gym.make('Walker2d-v4', render_mode='rgb_array')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG.load('models/{}'.format(run_name), env=env)
    
    data = evaluate_policy(model, model.get_env(), n_eval_episodes=100, return_episode_rewards=True)

    print('Mean reward:', mean(data[0]))
    print('Mean episode length:', mean(data[1]))

    env.close()

if __name__ == "__main__":
  	main()