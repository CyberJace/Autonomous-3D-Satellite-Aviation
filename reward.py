import numpy as np

def compute_spacecraft_reward(state, action, prev_action, goal_pos, obstacles):
    """
    Advanced Reward Function for Autonomous Spacecraft Navigation (ABV).
    
    Args:
        state: dict containing position [x,y,z], velocity [vx,vy,vz], fuel mass.
        action: [ax, ay, az, mode_selector] (continuous thrust vector + mode switch flag).
        prev_action: Action taken at previous timestep.
        goal_pos: Target [x,y,z] coordinates.
        obstacles: List of dicts [{'pos': [x,y,z], 'radius': r}, ...]
    """
    pos = np.array(state['pos'])
    vel = np.array(state['vel'])
    fuel = state['fuel']
    
    # 1. Target Proximity Reward (Progress shaping)
    dist_to_goal = np.linalg.norm(pos - goal_pos)
    progress_reward = 10.0 / (1.0 + dist_to_goal)
    
    # 2. Obstacle Avoidance Penalty
    collision_penalty = 0.0
    proximity_penalty = 0.0
    for obs in obstacles:
        obs_pos = np.array(obs['pos'])
        obs_r = obs['radius']
        d = np.linalg.norm(pos - obs_pos)
        
        if d < obs_r:
            collision_penalty -= 500.0 # Fatal collision
        elif d < (obs_r + 3.5): # Safety buffer zone
            proximity_penalty -= (3.5 - d) * 15.0

    # 3. Propulsion & Mode-Switching Cost (ABV Multi-Mode Management)
    thrust_magnitude = np.linalg.norm(action[:3])
    mode_switch = abs(action[3] - prev_action[3]) # Penalty for toggling thruster modes frequently
    
    # High-thrust mode vs Low-thrust ion mode fuel consumption penalties
    is_high_thrust = action[3] > 0.5
    fuel_burn_cost = thrust_magnitude * (0.05 if is_high_thrust else 0.01)
    chatter_penalty = mode_switch * 2.0
    
    # 4. Total Reward Aggregation
    total_reward = (
        progress_reward 
        + collision_penalty 
        + proximity_penalty 
        - fuel_burn_cost 
        - chatter_penalty
    )
    
    # Terminal bonus if goal reached
    terminated = dist_to_goal < 1.0
    if terminated:
        total_reward += 200.0

    return total_reward, terminated
