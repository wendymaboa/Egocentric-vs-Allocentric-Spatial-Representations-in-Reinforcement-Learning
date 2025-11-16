"""
Evaluate GoToObject models with added key as distractor
Tests if agents trained on 2 objects can handle 3 objects (2 targets + 1 key)
"""
import numpy as np
import pandas as pd
import argparse
import os
import gym
import gym_minigrid
from gym_minigrid.minigrid import Key

from stable_baselines3 import PPO, DQN
import envs.envs
from envs.wrappers import *


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    
    def observation(self, obs):
        return obs['image']


class AddKeyWrapper(gym.Wrapper):
    """Add a key as distractor object in GoToObject environment"""
    def __init__(self, env, key_color='yellow'):
        super().__init__(env)
        self.key_color = key_color
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Add a key to the environment
        # Find empty position
        empty_positions = []
        for i in range(1, self.unwrapped.width - 1):
            for j in range(1, self.unwrapped.height - 1):
                cell = self.unwrapped.grid.get(i, j)
                if cell is None:
                    empty_positions.append((i, j))
        
        if empty_positions:
            # Place key at random empty position
            key_pos = empty_positions[np.random.randint(len(empty_positions))]
            key = Key(self.key_color)
            self.unwrapped.grid.set(*key_pos, key)
            
            # Regenerate observation
            if hasattr(self.unwrapped, 'gen_obs'):
                obs = self.unwrapped.gen_obs()
        
        return obs


def evaluate_model(model, env, agent_type, n_episodes=100):
    """Evaluate model over n episodes"""
    successes = 0
    total_reward = 0
    total_steps = 0
    episode_rewards = []
    episode_steps = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done and ep_steps < 200:
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
        
        if ep_reward > 0:
            successes += 1
        
        total_reward += ep_reward
        total_steps += ep_steps
    
    return {
        'success_rate': successes / n_episodes,
        'avg_return': total_reward / n_episodes,
        'std_return': np.std(episode_rewards),
        'avg_steps': total_steps / n_episodes,
        'std_steps': np.std(episode_steps)
    }


def evaluate_with_key(env_key, agent_type, obs_type, seeds=[0, 1, 2], n_episodes=100, add_key=True):
    """Evaluate GoToObject models with optional key distractor"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {agent_type.upper()} - {obs_type}")
    print(f"Environment: {env_key}" + (" + KEY" if add_key else ""))
    print(f"{'='*80}\n")
    
    results = []
    
    for seed in seeds:
        model_path = f'models/{env_key}_{agent_type}_{obs_type}_seed{seed}.zip'
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping seed {seed}")
            continue
        
        print(f"Seed {seed}: Evaluating...")
        
        # Load model
        if agent_type == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Create environment
        env = gym.make(env_key)
        
        # Add key if requested
        if add_key:
            env = AddKeyWrapper(env, key_color='yellow')
        
        # Apply observation wrappers
        if obs_type == 'egocentric':
            env = FullyObsWrapper(env, egocentric=True)
        else:
            env = FullyObsWrapper(env, egocentric=False)
        
        env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
        env = FlattenObsWrapper(env)
        
        # Evaluate
        metrics = evaluate_model(model, env, agent_type, n_episodes)
        
        results.append({
            'seed': seed,
            'success_rate': metrics['success_rate'],
            'avg_return': metrics['avg_return'],
            'std_return': metrics['std_return'],
            'avg_steps': metrics['avg_steps']
        })
        
        print(f"  Success: {metrics['success_rate']*100:.1f}%, "
              f"Return: {metrics['avg_return']:.3f}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_key', default='MiniGrid-GoToObject-6x6-N2-v0')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', default='results/gotoobject_key_generalization.csv')
    
    args = parser.parse_args()
    
    configs = [
        ('ppo', 'egocentric'),
        ('ppo', 'allocentric'),
        ('dqn', 'egocentric'),
        ('dqn', 'allocentric')
    ]
    
    all_results = []
    
    print(f"\n{'='*80}")
    print("GOTOOBJECT GENERALIZATION TEST: Adding Key Distractor")
    print(f"{'='*80}\n")
    
    for agent_type, obs_type in configs:
        # Test 1: Standard (no key)
        print(f"\n--- Testing WITHOUT key ---")
        results_standard = evaluate_with_key(
            args.env_key, agent_type, obs_type, 
            args.seeds, args.n_episodes, add_key=False
        )
        
        # Test 2: With key
        print(f"\n--- Testing WITH key distractor ---")
        results_with_key = evaluate_with_key(
            args.env_key, agent_type, obs_type,
            args.seeds, args.n_episodes, add_key=True
        )
        
        if results_standard and results_with_key:
            # Calculate statistics
            success_standard = [r['success_rate'] for r in results_standard]
            success_with_key = [r['success_rate'] for r in results_with_key]
            
            mean_standard = np.mean(success_standard)
            mean_with_key = np.mean(success_with_key)
            drop = mean_standard - mean_with_key
            
            all_results.append({
                'Algorithm': agent_type.upper(),
                'Observation': obs_type.capitalize(),
                'Standard Success (%)': f"{mean_standard*100:.1f}",
                'With Key Success (%)': f"{mean_with_key*100:.1f}",
                'Performance Drop (%)': f"{drop*100:.1f}",
                'Seeds Standard': success_standard,
                'Seeds With Key': success_with_key
            })
    
    # Create summary table
    if all_results:
        summary_rows = []
        for result in all_results:
            summary_rows.append({
                'Algorithm': result['Algorithm'],
                'Observation': result['Observation'],
                'Standard (%)': result['Standard Success (%)'],
                'With Key (%)': result['With Key Success (%)'],
                'Drop (%)': result['Performance Drop (%)']
            })
        
        df = pd.DataFrame(summary_rows)
        
        print(f"\n{'='*80}")
        print("RESULTS: GoToObject with Key Distractor")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}\n")
        
        # Interpretation
        print("INTERPRETATION:")
        print("- Small drop (<10%): Good generalization to new objects")
        print("- Large drop (>30%): Overfitting to specific object configurations")
        print("- Perspective comparison: Does ego or allo handle distractors better?")


if __name__ == '__main__':
    main()