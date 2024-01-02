import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import gym.utils.seeding


class RaceTrackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RaceTrackEnv, self).__init__()
        self.track_width = 20
        self.field_size = (1000, 1000)  # Field size in pixels
        self.oval_center = (500, 500)
        self.oval_axes = (300, 400)
        self.car_size = (10, 15)  # Car size in pixels
        self.car_speed = 5  # Constant speed
        self.car_angle = 0  # Initial angle in degrees
        self.turn_angle = 5  # Angle by which the car turns for each action
        self.action_space = spaces.Discrete(2)  # actions: left and right
        self.observation_space = spaces.Box(low=np.array([0, 0, -360]), high=np.array([1000, 1000, 360]), dtype=np.float32)

        self.fig, self.ax = plt.subplots()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _get_observation(self):
    # Construct and return the observation
    # Example: if observation is just the car's position and angle
       return np.array(self.car_position + [self.car_angle])

    def reset(self, **kwargs):  # Accept arbitrary keyword arguments
        self.car_position = [self.oval_center[0], self.oval_center[1] - self.oval_axes[1] + self.track_width]
        self.car_angle = 0

        initial_observation = self._get_observation()
        return initial_observation  # This should only be the initial observation

    def step(self, action):
        if action == 0:  # Turn left
            self.car_angle -= self.turn_angle
        elif action == 1:  # Turn right
            self.car_angle += self.turn_angle

        dx = self.car_speed * np.cos(np.radians(self.car_angle))
        dy = self.car_speed * np.sin(np.radians(self.car_angle))
        self.car_position[0] += dx
        self.car_position[1] += dy

        done = self._is_off_track()
        reward = -1.0 if done else 1.0  # Reward is -1 for being off-track, 1 for staying on track

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            self.ax.clear()
            self.ax.set_xlim(0, self.field_size[0])
            self.ax.set_ylim(0, self.field_size[1])

            # Draw the track
            outer_ellipse = Ellipse(xy=self.oval_center, width=2*(self.oval_axes[0] + self.track_width/2), height=2*(self.oval_axes[1] + self.track_width/2), edgecolor='blue', fill=False)
            inner_ellipse = Ellipse(xy=self.oval_center, width=2*(self.oval_axes[0] - self.track_width/2), height=2*(self.oval_axes[1] - self.track_width/2), edgecolor='blue', fill=False)
            self.ax.add_patch(outer_ellipse)
            self.ax.add_patch(inner_ellipse)

            # Draw the car
            car_x, car_y = self.car_position
            car_rect = Rectangle((car_x - self.car_size[0] / 2, car_y - self.car_size[1] / 2), self.car_size[0], self.car_size[1], color='red', angle=self.car_angle)
            self.ax.add_patch(car_rect)

            plt.pause(0.01)  # pause to update the plot

    def close(self):
        plt.close()

    def _is_off_track(self):
        x, y = self.car_position
        cx, cy = self.oval_center
        a, b = self.oval_axes

        position = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2
        return position > 1  # True if the car is off track

    def _get_observation(self):
        # Return the car's x, y coordinates and heading
        return np.array(self.car_position + [self.car_angle])

# Example usage:
    
# env = RaceTrackEnv()
# env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         env.reset()
# env.close()
    
# debugging test:
    
env = RaceTrackEnv()
obs = env.reset()
print("Initial Observation:", obs)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Observation:", obs, "Reward:", reward, "Done:", done)

exit()

#training example    
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Create the Gym environment
env = RaceTrackEnv()
monitored_env = Monitor(env)

# Wrap it if necessary (vectorized environments allow for parallel computation)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Instantiate the agent
model = DQN('MlpPolicy', vec_env, verbose=1, learning_rate=1e-4, buffer_size=10000, learning_starts=1000, batch_size=32)

# Train the agent
model.learn(total_timesteps=int(1e5))

# Save the model
model.save("dqn_racetrack")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, monitored_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Remember to close the environment when done
monitored_env.close()
env.close()


#testing example:

from stable_baselines3 import DQN
import matplotlib.pyplot as plt

# Load the trained model
model = DQN.load("dqn_racetrack")

# Create and wrap your environment
env = RaceTrackEnv()
monitored_env = Monitor(env)

# Run the model in the environment
obs = monitored_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = monitored_env.step(action)
    monitored_env.render()
    if done:
        obs = monitored_env.reset()

# Close the environment
monitored_env.close()