import gymnasium as gym
from gymnasium.wrappers import HumanRendering

def main():
	env = HumanRendering(gym.make('Walker2d-v4', render_mode='rgb_array'))
	observation, info = env.reset()

	for _ in range(10000):
		action = env.action_space.sample()  # agent policy that uses the observation and info
		observation, reward, terminated, truncated, info = env.step(action)
		env.render()

	env.close()

if __name__ == "__main__":
  	main()