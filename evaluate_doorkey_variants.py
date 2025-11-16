"""
FIXED: Evaluate DoorKey models on modified environments
Now supports --env_key argument and multiple seeds
"""
import numpy as np
import pandas as pd
import argparse
import os
import gym
import gym_minigrid
from gym_minigrid.minigrid import Key, Door

from stable_baselines3 import PPO, DQN
import envs.envs
from envs.wrappers import *
from train_sb3 import FlattenObsWrapper, GymnasiumWrapper, Gym21CompatibilityWrapper


class RandomDoorKeyWrapper(gym.Wrapper):
    """Randomizes key and door colors in DoorKey"""
    def __init__(self, env, random_colors=True):
        super().__init__(env)
        self.random_colors = random_colors
        self.colors = ['red', 'green', 'blue', 'purple', 'yellow']
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        if not self.random_colors:
            return obs
        
        # Find and modify key and door colors
        for i in range(self.unwrapped.width):
            for j in range(self.unwrapped.height):
                cell = self.unwrapped.grid.get(i, j)
                
                if isinstance(cell, Key):
                    # Change key color
                    new_color = np.random.choice(self.colors)
                    new_key = Key(new_color)
                    self.unwrapped.grid.set(i, j, new_key)
                    
                elif isinstance(cell, Door):
                    # Change door color to match key
                    new_color = np.random.choice(self.colors)
                    new_door = Door(new_color, is_locked=True)
                    self.unwrapped.grid.set(i, j, new_door)
        
        # Regenerate observation
        obs = self.unwrapped.gen_obs()
        return obs


def evaluate_model(model, env, n_episodes=100, agent_type='ppo'):
    """Evaluate model over n episodes with detailed logging"""
    successes = 0
    total_reward = 0
    total_steps = 0
    episode_rewards = []
    episode_steps = []
    
    # Track what actually happens
    completed_task = 0
    timed_out = 0
    
    for ep in range(n_episodes):
        # Handle new gym API returning (obs, info) tuple
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        done = False
        ep_reward = 0
        ep_steps = 0
        max_reward_in_episode = 0
        
        while not done and ep_steps < 200:
            if agent_type == 'dqn':
                # DQN expects channels-first
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _ = model.predict(obs_transposed, deterministic=False)
            else:
                # PPO expects channels-last (which is what we have)
                action, _ = model.predict(obs, deterministic=True)
            
            # Handle new gym API returning 5 values
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            ep_reward += reward
            ep_steps += 1
            
            if reward > max_reward_in_episode:
                max_reward_in_episode = reward
        
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        
        # Detailed success tracking
        if done and ep_reward > 0.5:  # Successfully reached goal
            successes += 1
            completed_task += 1
        else:  # Episode timed out or failed
            timed_out += 1
        
        # Print first 5 episodes for diagnosis
        if ep < 5:
            print(f"    Ep {ep}: steps={ep_steps:3d}, reward={ep_reward:.3f}, "
                  f"done={done}, max_single_reward={max_reward_in_episode:.3f}")
        
        total_reward += ep_reward
        total_steps += ep_steps
    
    return {
        'success_rate': successes / n_episodes,
        'avg_return': total_reward / n_episodes,
        'std_return': np.std(episode_rewards),
        'avg_steps': total_steps / n_episodes,
        'std_steps': np.std(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'completed_task': completed_task,
        'timed_out': timed_out
    }


def create_eval_env(env_key, obs_type, random_colors=False):
    """
    Create environment with EXACT same wrappers as training
    This is critical for consistent evaluation!
    """
    env = gym.make(env_key)
    
    # Apply random color wrapper FIRST (before other wrappers)
    if random_colors:
        env = RandomDoorKeyWrapper(env, random_colors=True)
    
    # Apply observation wrappers in same order as training
    if obs_type == 'egocentric':
        env = FullyObsWrapper(env, egocentric=True)
    else:
        env = FullyObsWrapper(env, egocentric=False)
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
    env = FlattenObsWrapper(env)
    env = GymnasiumWrapper(env)
    env = Gym21CompatibilityWrapper(env)
    
    return env


def evaluate_multiple_seeds(env_key, agent_type, obs_type, seeds, n_episodes, random_colors):
    """Evaluate models from multiple seeds"""
    all_results = []
    
    for seed in seeds:
        model_path = f'models/{env_key}_{agent_type}_{obs_type}_seed{seed}.zip'
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}, skipping seed {seed}...")
            continue
        
        print(f"  Evaluating Seed {seed}...")
        
        # Load model
        if agent_type == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Create environment
        env = create_eval_env(env_key, obs_type, random_colors)
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes, agent_type)
        all_results.append(results)
        
        print(f"    Success Rate: {results['success_rate']*100:.1f}% | "
              f"Return: {results['avg_return']:.3f} ± {results['std_return']:.3f}")
    
    if not all_results:
        return None
    
    # Aggregate results across seeds
    return {
        'mean_success_rate': np.mean([r['success_rate'] for r in all_results]),
        'std_success_rate': np.std([r['success_rate'] for r in all_results]),
        'mean_return': np.mean([r['avg_return'] for r in all_results]),
        'std_return': np.std([r['avg_return'] for r in all_results]),
        'mean_steps': np.mean([r['avg_steps'] for r in all_results]),
        'std_steps': np.std([r['avg_steps'] for r in all_results]),
        'num_seeds': len(all_results)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_key', default='MiniGrid-DoorKey-5x5-v0', 
                        help='Environment to evaluate on')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes per evaluation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Seeds to evaluate')
    parser.add_argument('--output', default='results/doorkey_variants.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    model_configs = [
        ('ppo', 'allocentric'),
        ('ppo', 'egocentric'),
        ('dqn', 'allocentric'),
        ('dqn', 'egocentric'),
    ]
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"DOORKEY RANDOM COLOR GENERALIZATION TEST")
    print(f"Environment: {args.env_key}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.n_episodes}")
    print(f"{'='*80}\n")
    
    for agent_type, obs_type in model_configs:
        print(f"\n{agent_type.upper()} - {obs_type}:")
        
        # Test 1: Standard environment
        print(f"  Testing on STANDARD colors...")
        metrics_standard = evaluate_multiple_seeds(
            args.env_key, agent_type, obs_type, args.seeds, args.n_episodes, random_colors=False
        )
        
        if metrics_standard is None:
            print(f"  No models found for {agent_type}-{obs_type}, skipping...")
            continue
        
        print(f"  Standard: {metrics_standard['mean_success_rate']*100:.1f}% ± {metrics_standard['std_success_rate']*100:.1f}%")
        
        # Test 2: Random colors
        print(f"  Testing on RANDOM colors...")
        metrics_random = evaluate_multiple_seeds(
            args.env_key, agent_type, obs_type, args.seeds, args.n_episodes, random_colors=True
        )
        
        if metrics_random is None:
            continue
        
        print(f"  Random: {metrics_random['mean_success_rate']*100:.1f}% ± {metrics_random['std_success_rate']*100:.1f}%")
        
        # Calculate drop
        drop = metrics_standard['mean_success_rate'] - metrics_random['mean_success_rate']
        
        results.append({
            'Algorithm': agent_type.upper(),
            'Observation': obs_type.capitalize(),
            'Standard Success (%)': f"{metrics_standard['mean_success_rate']*100:.1f} ± {metrics_standard['std_success_rate']*100:.1f}",
            'Random Colors Success (%)': f"{metrics_random['mean_success_rate']*100:.1f} ± {metrics_random['std_success_rate']*100:.1f}",
            'Performance Drop (%)': f"{drop*100:.1f}",
            'Standard Return': f"{metrics_standard['mean_return']:.3f} ± {metrics_standard['std_return']:.3f}",
            'Random Return': f"{metrics_random['mean_return']:.3f} ± {metrics_random['std_return']:.3f}",
            'Num Seeds': metrics_standard['num_seeds']
        })
    
    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        
        print(f"\n{'='*80}")
        print("DOORKEY RANDOM COLOR RESULTS (Aggregated across seeds)")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        print(f"Results saved to: {args.output}")
    else:
        print("\nNo results to save!")