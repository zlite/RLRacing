import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import gym.utils.seeding
from scipy.special import binom
import matplotlib.pyplot as plt


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
    



    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

    def bezier(points, num=200):
        N = len(points)
        t = np.linspace(0, 1, num=num)
        curve = np.zeros((num, 2))
        for i in range(N):
            curve += np.outer(bernstein(N - 1, i, t), points[i])
        return curve

    class Segment():
        def __init__(self, p1, p2, angle1, angle2, **kw):
            self.p1 = p1; self.p2 = p2
            self.angle1 = angle1; self.angle2 = angle2
            self.numpoints = kw.get("numpoints", 100)
            r = kw.get("r", 0.3)
            d = np.sqrt(np.sum((self.p2-self.p1)**2))
            self.r = r*d
            self.p = np.zeros((4,2))
            self.p[0,:] = self.p1[:]
            self.p[3,:] = self.p2[:]
            self.calc_intermediate_points(self.r)

        def calc_intermediate_points(self,r):
            self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                        self.r*np.sin(self.angle1)])
            self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                        self.r*np.sin(self.angle2+np.pi)])
            self.curve = bezier(self.p,self.numpoints)


    def get_curve(points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def ccw_sort(p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def get_bezier_curve(a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points. 
        *rad* is a number between 0 and 1 to steer the distance of
            control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
            edgy=0 is smoothest."""
        p = np.arctan(edgy)/np.pi+.5
        a = ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:,1],d[:,0])
        f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang,1)
        ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = get_curve(a, r=rad, method="var")
        x,y = c.T
        return x,y, a

    def scale_bezier_curves(control_points, scaling_factor):
        # Convert control points to a NumPy array if not already
        control_points_np = np.array(control_points)
        
        # Calculate the center of the shape
        center = control_points_np.mean(axis=0)
        
        # Scale control points
        scaled_control_points = center + (control_points_np - center) * scaling_factor
        
        return scaled_control_points

    def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        a = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            return a*scale
        else:
            return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
        
    def make_road(self) -> None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        rad = 0.2
        edgy = 0.05

        a = get_random_points(n=7, scale=1)
        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
        plt.plot(x,y, linewidth=15)
        # x1,y1, _ = get_bezier_curve(b,rad=rad, edgy=edgy)
        # plt.plot(x1,y1)
        # x2,y2, _ = get_bezier_curve(c,rad=rad, edgy=edgy)
        # plt.plot(x2,y2)
        plt.show()

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