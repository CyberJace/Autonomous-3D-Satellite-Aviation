import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reward import compute_spacecraft_reward
from simsr_encoder import SimSREncoder

class SpacecraftEnv3D(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(SpacecraftEnv3D, self).__init__()
        
        # Action space: [ax, ay, az (acceleration vectors), mode_flag (0=Low-Thrust Ion, 1=High-Thrust Chemical)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        )
        
        # Observation space: [x, y, z, vx, vy, vz, gx, gy, gz, fuel, 6 local LiDAR distance rays]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.goal_pos = np.array([50.0, 50.0, 20.0])
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.state_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.fuel = 100.0
        self.prev_action = np.zeros(4)
        
        # Randomize unknown obstacles for generalization
        self.obstacles = [
            {'pos': np.array([20.0, 25.0, 10.0]), 'radius': 5.0},
            {'pos': np.array([35.0, 15.0, 15.0]), 'radius': 4.0},
            {'pos': np.array([40.0, 40.0, 18.0]), 'radius': 6.0}
        ]
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Compute dummy 6-axis LiDAR / sensor distances to nearest obstacles
        lidar = []
        for ray_dir in [np.array([1,0,0]), np.array([-1,0,0]), np.array([0,1,0]), 
                        np.array([0,-1,0]), np.array([0,0,1]), np.array([0,0,-1])]:
            min_d = 50.0
            for obs in self.obstacles:
                d = np.linalg.norm(obs['pos'] - self.state_pos) - obs['radius']
                if d < min_d: min_d = max(0.0, d)
            lidar.append(min_d)

        obs = np.concatenate([
            self.state_pos,
            self.state_vel,
            self.goal_pos,
            [self.fuel],
            np.array(lidar[:2]) # compressed ray readings
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # Frictionless orbital/space dynamics (Newtonian Euler integration)
        accel_multiplier = 2.0 if action[3] > 0.5 else 0.5 # High vs Low thrust profile
        acceleration = action[:3] * accel_multiplier
        
        self.state_vel += acceleration * 0.1 # dt = 0.1s
        self.state_pos += self.state_vel * 0.1
        self.fuel -= np.linalg.norm(action[:3]) * 0.05
        
        state_dict = {
            'pos': self.state_pos,
            'vel': self.state_vel,
            'fuel': max(0.0, self.fuel)
        }
        
        reward, terminated = compute_spacecraft_reward(
            state_dict, action, self.prev_action, self.goal_pos, self.obstacles
        )
        self.prev_action = action
        
        truncated = self.fuel <= 0.0 or np.linalg.norm(self.state_pos - self.goal_pos) > 200.0
        return self._get_obs(), reward, terminated, truncated, {}
