"""
Script to visualize and diagnose suspicious model behaviors
Specifically targets:
1. PPO-Egocentric Empty-8x8 failure (0% success)
2. DoorKey performance inconsistencies
3. Random start generalization issues
"""
import numpy as np
import os
import argparse
import gym
import gym_minigrid
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import PPO, DQN

import envs.envs
from envs.wrappers import *


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    
    def observation(self, obs):
        return obs['image']


class RandomStartWrapper(gym.Wrapper):
    """Places agent at random position on reset"""
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Get all empty positions
        empty_positions = []
        for i in range(1, self.unwrapped.width - 1):
            for j in range(1, self.unwrapped.height - 1):
                cell = self.unwrapped.grid.get(i, j)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    empty_positions.append((i, j))
        
        # Place agent at random empty position
        if empty_positions:
            new_pos = empty_positions[np.random.randint(len(empty_positions))]
            self.unwrapped.agent_pos = new_pos
            self.unwrapped.agent_dir = np.random.randint(0, 4)
            
            # Regenerate observation
            if hasattr(self.unwrapped, 'gen_obs'):
                obs = self.unwrapped.gen_obs()
        
        return obs


def fig_image(fig):
    """Convert matplotlib figure to image array"""
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def diagnose_model(model, env, agent_type, observation_type, n_episodes=100, 
                   save_video=False, video_name="diagnosis", max_video_episodes=5):
    """
    Diagnose model behavior with detailed logging and optional video
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSING MODEL: {agent_type.upper()} - {observation_type}")
    print(f"{'='*80}\n")
    
    # Statistics
    episode_rewards = []
    episode_steps = []
    episode_outcomes = {'success': 0, 'timeout': 0, 'other': 0}
    
    # Track action distribution
    action_counts = {}
    
    # Track positions visited (for navigation tasks)
    positions_visited = []
    
    # Video frames
    images = [] if save_video else None
    video_episode_count = 0
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        ep_actions = []
        ep_positions = []
        
        # Print detailed info for first 10 episodes
        verbose = (ep < 10)
        
        if verbose:
            print(f"Episode {ep+1}:")
            print(f"  Initial agent position: {env.unwrapped.agent_pos}")
            print(f"  Initial agent direction: {env.unwrapped.agent_dir}")
        
        while not done and step < 200:
            # Get action
            if agent_type == 'dqn':
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _ = model.predict(obs_transposed, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            ep_actions.append(action)
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Record position
            ep_positions.append(tuple(env.unwrapped.agent_pos))
            
            # Save video frames for failed episodes
            if save_video and video_episode_count < max_video_episodes:
                image_global = env.render("rgb_array", highlight=True)
                image_agent = obs
                
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
                axs[0].set_title("Global View", fontsize=16)
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].imshow(image_global)
                
                axs[1].set_title(f"Agent View ({observation_type})", fontsize=16)
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                axs[1].imshow(image_agent)
                
                action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
                action_name = action_names[action] if action < len(action_names) else f'Action{action}'
                
                fig.suptitle(f"Ep {ep+1} | Step {step} | Action: {action_name} | Reward: {episode_reward:.2f}", 
                            fontsize=14, y=0.95)
                
                images.append(fig_image(fig))
            
            # Step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        # Categorize outcome
        if done and episode_reward > 0:
            episode_outcomes['success'] += 1
            outcome_str = "SUCCESS"
        elif step >= 200:
            episode_outcomes['timeout'] += 1
            outcome_str = "TIMEOUT"
            if save_video and video_episode_count < max_video_episodes:
                video_episode_count += 1
        else:
            episode_outcomes['other'] += 1
            outcome_str = "FAILED"
            if save_video and video_episode_count < max_video_episodes:
                video_episode_count += 1
        
        # Calculate unique positions visited
        unique_positions = len(set(ep_positions))
        positions_visited.append(unique_positions)
        
        if verbose:
            # Analyze action distribution for this episode
            from collections import Counter
            action_dist = Counter(ep_actions)
            print(f"  Outcome: {outcome_str}")
            print(f"  Steps: {step}, Reward: {episode_reward:.3f}")
            print(f"  Unique positions visited: {unique_positions}")
            print(f"  Action distribution: {dict(action_dist)}")
            
            # Check for suspicious patterns
            if len(set(ep_actions)) <= 2:
                print(f"  ⚠️  WARNING: Agent only using {len(set(ep_actions))} different actions!")
            
            if unique_positions < 5 and step > 50:
                print(f"  ⚠️  WARNING: Agent not exploring (only {unique_positions} positions in {step} steps)!")
            
            print()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes: {n_episodes}")
    print(f"\nOutcomes:")
    print(f"  Success: {episode_outcomes['success']} ({episode_outcomes['success']/n_episodes*100:.1f}%)")
    print(f"  Timeout: {episode_outcomes['timeout']} ({episode_outcomes['timeout']/n_episodes*100:.1f}%)")
    print(f"  Other: {episode_outcomes['other']} ({episode_outcomes['other']/n_episodes*100:.1f}%)")
    
    print(f"\nPerformance:")
    print(f"  Mean reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Mean steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    print(f"  Mean unique positions: {np.mean(positions_visited):.1f} ± {np.std(positions_visited):.1f}")
    
    print(f"\nAction Distribution (across all episodes):")
    action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        action_name = action_names[action] if action < len(action_names) else f'Action{action}'
        print(f"  {action_name}: {count} ({count/total_actions*100:.1f}%)")
    
    # Identify suspicious behaviors
    print(f"\n{'='*80}")
    print(f"SUSPICIOUS BEHAVIOR DETECTION")
    print(f"{'='*80}")
    
    suspicious = []
    
    if episode_outcomes['success'] == 0:
        suspicious.append("🚨 CRITICAL: 0% success rate - model completely failed!")
    
    if episode_outcomes['timeout'] > n_episodes * 0.8:
        suspicious.append(f"⚠️  {episode_outcomes['timeout']/n_episodes*100:.1f}% episodes timed out - agent not reaching goal")
    
    if len(action_counts) <= 2:
        suspicious.append(f"⚠️  Agent only uses {len(action_counts)} different actions - policy collapsed!")
    
    if max(action_counts.values()) / total_actions > 0.8:
        dominant_action = max(action_counts, key=action_counts.get)
        action_name = action_names[dominant_action] if dominant_action < len(action_names) else f'Action{dominant_action}'
        suspicious.append(f"⚠️  One action dominates: {action_name} ({action_counts[dominant_action]/total_actions*100:.1f}%)")
    
    if np.mean(positions_visited) < 10 and 'Empty' in str(env.unwrapped):
        suspicious.append(f"⚠️  Poor exploration: only visiting {np.mean(positions_visited):.1f} positions on average")
    
    if suspicious:
        for issue in suspicious:
            print(issue)
    else:
        print("✓ No obvious suspicious behaviors detected")
    
    print(f"{'='*80}\n")
    
    # Save video if requested
    if save_video and images:
        os.makedirs("diagnosis_videos", exist_ok=True)
        output_path = f"diagnosis_videos/{video_name}.gif"
        imageio.mimsave(output_path, images, fps=3, loop=0)
        print(f"Diagnosis video saved to: {output_path}\n")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_outcomes': episode_outcomes,
        'action_counts': action_counts,
        'positions_visited': positions_visited,
        'suspicious_behaviors': suspicious
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose suspicious model behaviors')
    parser.add_argument('--env_key', default="MiniGrid-Empty-8x8-v0")
    parser.add_argument('--agent_type', default="ppo", choices=["ppo", "dqn"])
    parser.add_argument('--egocentric', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--random_start', action='store_true', default=False,
                       help='Test with random starting positions')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--model_path', default=None)
    
    args = parser.parse_args()
    
    # Create environment
    env = gym.make(args.env_key)
    
    # Add random start wrapper if requested
    if args.random_start:
        env = RandomStartWrapper(env)
        print("⚠️  Using RANDOM START positions")
    
    # Apply observation wrappers
    observation_type = "egocentric" if args.egocentric else "allocentric"
    if args.egocentric:
        env = FullyObsWrapper(env, egocentric=True)
    else:
        env = FullyObsWrapper(env, egocentric=False)
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
    env = FlattenObsWrapper(env)
    
    # Determine model path
    if args.model_path:
        path = args.model_path
    else:
        if args.seed is not None:
            path = f'models/{args.env_key}_{args.agent_type}_{observation_type}_seed{args.seed}.zip'
        else:
            path = f'models/{args.env_key}_{args.agent_type}_{observation_type}.zip'
    
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        print("\nAvailable models:")
        if os.path.exists('models/'):
            for f in os.listdir('models/'):
                if f.endswith('.zip') and args.env_key in f:
                    print(f"  - {f}")
        exit(1)
    
    print(f"Loading model from: {path}\n")
    
    # Load model
    if args.agent_type == 'ppo':
        model = PPO.load(path)
    else:
        model = DQN.load(path)
    
    # Create video name
    random_str = "_random_start" if args.random_start else ""
    seed_str = f"_seed{args.seed}" if args.seed is not None else ""
    video_name = f"{args.env_key}_{args.agent_type}_{observation_type}{seed_str}{random_str}"
    
    # Run diagnosis
    results = diagnose_model(
        model, env, args.agent_type, observation_type,
        n_episodes=args.episodes,
        save_video=args.save_video,
        video_name=video_name
    )