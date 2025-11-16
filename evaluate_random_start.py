"""
Evaluate trained models on random starting positions
Tests generalization: trained on fixed start, evaluated on random start
Supports multiple seeds and entropy tracking (PPO only)
"""
import numpy as np
import pandas as pd
import argparse
import os
import gym
import gym_minigrid
import torch

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


def compute_ppo_entropy(model, obs):
    """Compute entropy for PPO policy"""
    try:
        # FIX: Transpose from (H, W, C) to (C, H, W) for PyTorch
        obs_transposed = np.transpose(obs, (2, 0, 1))
        obs_tensor = torch.as_tensor(obs_transposed).unsqueeze(0).to(model.device)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            entropy = distribution.entropy().mean().item()
        return entropy
    except Exception as e:
        print(f"Warning: Could not compute entropy: {e}")
        return None


def evaluate_model(model, env, n_episodes=100, agent_type='ppo'):
    """Evaluate model over n episodes with entropy tracking for PPO"""
    successes = 0
    total_reward = 0
    total_steps = 0
    episode_rewards = []
    episode_steps = []
    
    # Track entropy for PPO
    entropies = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_entropies = []
        
        while not done and ep_steps < 200:
            # Compute entropy for PPO before taking action
            if agent_type == 'ppo':
                entropy = compute_ppo_entropy(model, obs)
                if entropy is not None:
                    ep_entropies.append(entropy)
            
            if agent_type == 'dqn':
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _ = model.predict(obs_transposed, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
        
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        
        # Store mean entropy for this episode (PPO only)
        if agent_type == 'ppo' and ep_entropies:
            entropies.append(np.mean(ep_entropies))
        
        # Check actual goal completion
        if done and ep_reward > 0:
            successes += 1
        
        total_reward += ep_reward
        total_steps += ep_steps
    
    results = {
        'success_rate': successes / n_episodes,
        'avg_return': total_reward / n_episodes,
        'std_return': np.std(episode_rewards),
        'avg_steps': total_steps / n_episodes,
        'std_steps': np.std(episode_steps),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps
    }
    
    # Add entropy statistics for PPO
    if agent_type == 'ppo' and entropies:
        results['avg_entropy'] = np.mean(entropies)
        results['std_entropy'] = np.std(entropies)
    else:
        results['avg_entropy'] = None
        results['std_entropy'] = None
    
    return results


def evaluate_multiple_seeds(env_key, agent_type, obs_type, seeds=[0, 1, 2], n_episodes=100):
    """Evaluate models from multiple seeds on random start positions"""
    
    all_results = []
    
    for seed in seeds:
        model_path = f'models/{env_key}_{agent_type}_{obs_type}_seed{seed}.zip'
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping seed {seed}...")
            continue
        
        print(f"Evaluating {agent_type.upper()} - {obs_type} - Seed {seed} (Random Start)...")
        
        # Create environment WITH random start
        env = gym.make(env_key)
        env = RandomStartWrapper(env)  # Add random start wrapper
        
        if obs_type == 'egocentric':
            env = FullyObsWrapper(env, egocentric=True)
        else:
            env = FullyObsWrapper(env, egocentric=False)
        
        env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
        env = FlattenObsWrapper(env)
        
        # Load model
        if agent_type == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes, agent_type)
        all_results.append(results)
        
        entropy_str = ""
        if results['avg_entropy'] is not None:
            entropy_str = f" | Entropy: {results['avg_entropy']:.3f} ± {results['std_entropy']:.3f}"
        
        print(f"  Seed {seed}: Success Rate: {results['success_rate']*100:.1f}% | "
              f"Return: {results['avg_return']:.3f} ± {results['std_return']:.3f}{entropy_str}\n")
    
    if not all_results:
        return None
    
    # Aggregate results across seeds
    aggregated = {
        'mean_success_rate': np.mean([r['success_rate'] for r in all_results]),
        'std_success_rate': np.std([r['success_rate'] for r in all_results]),
        'mean_return': np.mean([r['avg_return'] for r in all_results]),
        'std_return_across_seeds': np.std([r['avg_return'] for r in all_results]),
        'mean_steps': np.mean([r['avg_steps'] for r in all_results]),
        'std_steps_across_seeds': np.std([r['avg_steps'] for r in all_results]),
        'num_seeds': len(all_results),
        'individual_results': all_results
    }
    
    # Add entropy aggregation for PPO
    if agent_type == 'ppo' and all_results[0]['avg_entropy'] is not None:
        aggregated['mean_entropy'] = np.mean([r['avg_entropy'] for r in all_results])
        aggregated['std_entropy_across_seeds'] = np.std([r['avg_entropy'] for r in all_results])
    else:
        aggregated['mean_entropy'] = None
        aggregated['std_entropy_across_seeds'] = None
    
    return aggregated


def create_comparison_table(results_dict, output_file='results/random_start_eval.csv'):
    """Create comparison table from results with entropy column"""
    rows = []
    
    for key, metrics in results_dict.items():
        env_name, agent_type, obs_type = key
        
        row = {
            'Environment': env_name,
            'Algorithm': agent_type.upper(),
            'Observation': obs_type.capitalize(),
            'Success Rate (%)': f"{metrics['mean_success_rate']*100:.1f} ± {metrics['std_success_rate']*100:.1f}",
            'Avg Return': f"{metrics['mean_return']:.3f} ± {metrics['std_return_across_seeds']:.3f}",
            'Avg Steps': f"{metrics['mean_steps']:.1f} ± {metrics['std_steps_across_seeds']:.1f}",
            'Avg Entropy': f"{metrics['mean_entropy']:.3f} ± {metrics['std_entropy_across_seeds']:.3f}" if metrics['mean_entropy'] is not None else 'N/A',
            'Num Seeds': metrics['num_seeds']
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Print formatted table
    print("\n" + "="*120)
    print("GENERALIZATION TEST: Random Starting Positions (Aggregated across seeds)")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_key', default="MiniGrid-Empty-5x5-v0")
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', default='results/random_start_eval.csv')
    
    args = parser.parse_args()
    
    # Models to evaluate
    model_configs = [
        ('ppo', 'allocentric'),
        ('ppo', 'egocentric'),
        ('dqn', 'allocentric'),
        ('dqn', 'egocentric'),
    ]
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"GENERALIZATION TEST: Trained on Fixed Start, Evaluated on Random Start")
    print(f"Environment: {args.env_key}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.n_episodes}")
    print(f"{'='*80}\n")
    
    for agent_type, obs_type in model_configs:
        aggregated = evaluate_multiple_seeds(
            args.env_key, 
            agent_type, 
            obs_type, 
            args.seeds, 
            args.n_episodes
        )
        
        if aggregated is not None:
            key = (args.env_key, agent_type, obs_type)
            results[key] = aggregated
            
            entropy_str = ""
            if aggregated['mean_entropy'] is not None:
                entropy_str = f"\n  Mean Entropy: {aggregated['mean_entropy']:.3f} ± {aggregated['std_entropy_across_seeds']:.3f}"
            
            print(f"\n{agent_type.upper()}-{obs_type} SUMMARY:")
            print(f"  Mean Success Rate: {aggregated['mean_success_rate']*100:.1f}% ± {aggregated['std_success_rate']*100:.1f}%")
            print(f"  Mean Return: {aggregated['mean_return']:.3f} ± {aggregated['std_return_across_seeds']:.3f}")
            print(f"  Mean Steps: {aggregated['mean_steps']:.1f} ± {aggregated['std_steps_across_seeds']:.1f}{entropy_str}")
            print(f"  ({aggregated['num_seeds']} seeds)")
    
    # Create comparison table
    if results:
        create_comparison_table(results, args.output)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo models found to evaluate!")