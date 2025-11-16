"""
Evaluate GoToObject agents on environments with MORE objects
Tests if agents can generalize to more complex object discrimination tasks
"""
import numpy as np
import pandas as pd
import argparse
import os
import gym
import gym_minigrid
from gym_minigrid.minigrid import Ball, Box, Key
import torch

from stable_baselines3 import PPO, DQN
import envs.envs
from envs.wrappers import *
from train_sb3 import FlattenObsWrapper, GymnasiumWrapper, Gym21CompatibilityWrapper


class MoreObjectsWrapper(gym.Wrapper):
    """Add more objects to GoToObject environment for harder discrimination"""
    def __init__(self, env, num_extra_objects=2):
        super().__init__(env)
        self.num_extra_objects = num_extra_objects
        self.object_types = [Ball, Box, Key]
        self.colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Add extra objects to the environment
        for _ in range(self.num_extra_objects):
            self._add_random_object()
        
        # Regenerate observation
        obs = self.unwrapped.gen_obs()
        return obs
    
    def _add_random_object(self):
        """Add a random object at a random empty position"""
        grid = self.unwrapped.grid
        
        # Find empty positions
        empty_positions = []
        for i in range(1, grid.width - 1):
            for j in range(1, grid.height - 1):
                if grid.get(i, j) is None and (i, j) != tuple(self.unwrapped.agent_pos):
                    empty_positions.append((i, j))
        
        if empty_positions:
            # Choose random position
            pos = empty_positions[np.random.randint(len(empty_positions))]
            
            # Choose random object type and color
            obj_type = np.random.choice(self.object_types)
            color = np.random.choice(self.colors)
            
            # Create and place object
            obj = obj_type(color)
            grid.set(pos[0], pos[1], obj)


def compute_ppo_entropy(model, obs):
    """Compute entropy for PPO policy"""
    try:
        obs_transposed = np.transpose(obs, (2, 0, 1))
        obs_tensor = torch.as_tensor(obs_transposed).unsqueeze(0).to(model.device)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            entropy = distribution.entropy().mean().item()
        return entropy
    except Exception as e:
        return None


def evaluate_model(model, env, n_episodes=100, agent_type='ppo'):
    """Evaluate model over n episodes"""
    successes = 0
    total_reward = 0
    total_steps = 0
    episode_rewards = []
    episode_steps = []
    entropies = []
    
    for ep in range(n_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        done = False
        ep_reward = 0
        ep_steps = 0
        ep_entropies = []
        
        while not done and ep_steps < 200:
            # Compute entropy for PPO
            if agent_type == 'ppo':
                entropy = compute_ppo_entropy(model, obs)
                if entropy is not None:
                    ep_entropies.append(entropy)
            
            if agent_type == 'dqn':
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _ = model.predict(obs_transposed, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            ep_reward += reward
            ep_steps += 1
        
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        
        if agent_type == 'ppo' and ep_entropies:
            entropies.append(np.mean(ep_entropies))
        
        if ep_reward > 0.5:
            successes += 1
        
        # Print first 5 episodes
        if ep < 5:
            print(f"    Ep {ep}: reward={ep_reward:.3f}, steps={ep_steps}")
        
        total_reward += ep_reward
        total_steps += ep_steps
    
    results = {
        'success_rate': successes / n_episodes,
        'avg_return': total_reward / n_episodes,
        'std_return': np.std(episode_rewards),
        'avg_steps': total_steps / n_episodes,
        'std_steps': np.std(episode_steps),
    }
    
    if agent_type == 'ppo' and entropies:
        results['avg_entropy'] = np.mean(entropies)
        results['std_entropy'] = np.std(entropies)
    
    return results


def create_eval_env(env_key, obs_type, num_extra_objects=0):
    """Create evaluation environment with optional extra objects"""
    env = gym.make(env_key)
    
    # Add more objects if requested
    if num_extra_objects > 0:
        env = MoreObjectsWrapper(env, num_extra_objects=num_extra_objects)
    
    # Apply observation wrappers
    if obs_type == 'egocentric':
        env = FullyObsWrapper(env, egocentric=True)
    else:
        env = FullyObsWrapper(env, egocentric=False)
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
    env = FlattenObsWrapper(env)
    env = GymnasiumWrapper(env)
    env = Gym21CompatibilityWrapper(env)
    
    return env


def evaluate_multiple_seeds(train_env, test_env, agent_type, obs_type, seeds, n_episodes, num_extra_objects):
    """Evaluate models from multiple seeds"""
    all_results = []
    
    for seed in seeds:
        # Models were trained on train_env
        model_path = f'models/{train_env}_{agent_type}_{obs_type}_seed{seed}.zip'
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}, skipping seed {seed}...")
            continue
        
        print(f"  Evaluating Seed {seed}...")
        
        # Load model
        if agent_type == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Create test environment
        env = create_eval_env(test_env, obs_type, num_extra_objects)
        
        # Evaluate
        results = evaluate_model(model, env, n_episodes, agent_type)
        all_results.append(results)
        
        entropy_str = ""
        if 'avg_entropy' in results:
            entropy_str = f" | Entropy: {results['avg_entropy']:.3f}"
        
        print(f"    Success: {results['success_rate']*100:.1f}% | "
              f"Return: {results['avg_return']:.3f}{entropy_str}")
    
    if not all_results:
        return None
    
    # Aggregate
    aggregated = {
        'mean_success_rate': np.mean([r['success_rate'] for r in all_results]),
        'std_success_rate': np.std([r['success_rate'] for r in all_results]),
        'mean_return': np.mean([r['avg_return'] for r in all_results]),
        'std_return': np.std([r['avg_return'] for r in all_results]),
        'mean_steps': np.mean([r['avg_steps'] for r in all_results]),
        'std_steps': np.std([r['avg_steps'] for r in all_results]),
        'num_seeds': len(all_results)
    }
    
    if 'avg_entropy' in all_results[0]:
        aggregated['mean_entropy'] = np.mean([r['avg_entropy'] for r in all_results])
        aggregated['std_entropy'] = np.std([r['avg_entropy'] for r in all_results])
    
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_env', default='MiniGrid-GoToObject-6x6-N2-v0',
                        help='Environment models were trained on')
    parser.add_argument('--test_envs', nargs='+', 
                        default=['MiniGrid-GoToObject-6x6-N2-v0'],
                        help='Environments to test on')
    parser.add_argument('--extra_objects', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='Number of extra objects to add (0 = baseline)')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', default='results/goto_more_objects.csv')
    
    args = parser.parse_args()
    
    model_configs = [
        ('ppo', 'allocentric'),
        ('ppo', 'egocentric'),
        ('dqn', 'allocentric'),
        ('dqn', 'egocentric'),
    ]
    
    all_results = []
    
    print(f"\n{'='*100}")
    print(f"GOTO OBJECT: MORE OBJECTS GENERALIZATION TEST")
    print(f"Trained on: {args.train_env}")
    print(f"Testing on: {args.test_envs}")
    print(f"Extra objects to add: {args.extra_objects}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*100}\n")
    
    # Test each combination
    for test_env in args.test_envs:
        for num_extra in args.extra_objects:
            test_name = f"{test_env} (+{num_extra} objects)" if num_extra > 0 else test_env
            
            print(f"\n{'='*100}")
            print(f"TESTING: {test_name}")
            print(f"{'='*100}")
            
            for agent_type, obs_type in model_configs:
                print(f"\n{agent_type.upper()}-{obs_type}:")
                
                metrics = evaluate_multiple_seeds(
                    args.train_env, test_env, agent_type, obs_type, 
                    args.seeds, args.n_episodes, num_extra
                )
                
                if metrics is None:
                    print(f"  No models found, skipping...")
                    continue
                
                print(f"  Aggregated: {metrics['mean_success_rate']*100:.1f}% ± {metrics['std_success_rate']*100:.1f}%")
                
                result_row = {
                    'Train Env': args.train_env,
                    'Test Env': test_env,
                    'Extra Objects': num_extra,
                    'Algorithm': agent_type.upper(),
                    'Observation': obs_type.capitalize(),
                    'Success Rate (%)': f"{metrics['mean_success_rate']*100:.1f} ± {metrics['std_success_rate']*100:.1f}",
                    'Avg Return': f"{metrics['mean_return']:.3f} ± {metrics['std_return']:.3f}",
                    'Avg Steps': f"{metrics['mean_steps']:.1f} ± {metrics['std_steps']:.1f}",
                    'Num Seeds': metrics['num_seeds']
                }
                
                if 'mean_entropy' in metrics:
                    result_row['Avg Entropy'] = f"{metrics['mean_entropy']:.3f} ± {metrics['std_entropy']:.3f}"
                
                all_results.append(result_row)
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        
        print(f"\n{'='*100}")
        print("FINAL RESULTS: MORE OBJECTS GENERALIZATION")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        print(f"{'='*100}\n")
        
        print(f"Results saved to: {args.output}")
        
        # Print summary comparison
        print(f"\n{'='*100}")
        print("PERFORMANCE DROP ANALYSIS")
        print(f"{'='*100}")
        
        baseline_df = df[df['Extra Objects'] == 0]
        for _, baseline in baseline_df.iterrows():
            algo = baseline['Algorithm']
            obs = baseline['Observation']
            baseline_sr = float(baseline['Success Rate (%)'].split('±')[0])
            
            print(f"\n{algo}-{obs}:")
            
            agent_df = df[(df['Algorithm'] == algo) & (df['Observation'] == obs)]
            for _, row in agent_df.iterrows():
                extra = row['Extra Objects']
                sr = float(row['Success Rate (%)'].split('±')[0])
                drop = baseline_sr - sr
                print(f"  +{extra} objects: {sr:.1f}% (drop: {drop:.1f}%)")
    else:
        print("\nNo results to save!")