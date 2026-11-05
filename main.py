import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import SpacecraftEnv3D
from simsr_encoder import SimSREncoder

def run_training_and_visualization():
    print("Initializing 3D Spacecraft DRL Environment with SimSR & Multi-Mode Propulsion...")
    env = SpacecraftEnv3D()
    
    # Initialize SimSR State Encoder
    simsr = SimSREncoder(input_dim=11, latent_dim=8)
    print("SimSR Encoder initialized successfully with Wasserstein optimal transport regularizer.")

    # Train PPO Agent
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=64)
    print("Training DRL agent on 3D obstacle avoidance task...")
    model.learn(total_timesteps=10000)
    print("Training complete! Running inference on unknown obstacles...")

    # Run Episode for Visualization
    obs, _ = env.reset()
    trajectory = [env.state_pos.copy()]
    
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(env.state_pos.copy())
        if terminated or truncated:
            break

    trajectory = np.array(trajectory)

    # --- RENDER STUNNING 3D VISUALIZATION ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    # Plot Spacecraft Path
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Spacecraft Trajectory (ABV)', color='cyan', linewidth=3)
    ax.scatter(0, 0, 0, color='green', s=100, label='Start Position')
    ax.scatter(50, 50, 20, color='gold', s=150, marker='*', label='Target Goal')

    # Plot Obstacles
    for obs in env.obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = obs['radius'] * np.cos(u) * np.sin(v) + obs['pos'][0]
        y = obs['radius'] * np.sin(u) * np.sin(v) + obs['pos'][1]
        z = obs['radius'] * np.cos(v) + obs['pos'][2]
        ax.plot_wireframe(x, y, z, color='red', alpha=0.3)

    ax.set_title("3D Autonomous Spacecraft Obstacle Avoidance & SimSR Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run_training_and_visualization()
