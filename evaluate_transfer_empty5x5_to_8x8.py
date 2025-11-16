"""
Evaluate transfer learning: Train on Empty-5x5, test on Empty-8x8
Tests if agents can generalize to larger environments
"""
import numpy as np
import pandas as pd
import argparse
import os
import gym
import gym_minigrid

from stable_baselines3 import PPO, DQN
import envs.envs
from envs.wrappers import *


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    
    def observation(self, obs):
        return obs['image']


def evaluate_model(model, env, agent_type, n_episodes=100):
    """Evaluate model on environment"""
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


def test_transfer(train_env, test_env, agent_type, obs_type, seeds=[0, 1, 2], n_episodes=100):
    """Test transfer from train_env to test_env"""
    
    print(f"\n{'='*80}")
    print(f"Transfer Test: {train_env} → {test_env}")
    print(f"Agent: {agent_type.upper()}, Observation: {obs_type}")
    print(f"{'='*80}\n")
    
    results = []
    
    for seed in seeds:
        # Load model trained on train_env
        model_path = f'models/{train_env}_{agent_type}_{obs_type}_seed{seed}.zip'
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping seed {seed}")
            continue
        
        print(f"Seed {seed}: Loading model from {model_path}")
        
        # Load model
        if agent_type == 'ppo':
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        
        # Create test environment
        env = gym.make(test_env)
        
        if obs_type == 'egocentric':
            env = FullyObsWrapper(env, egocentric=True)
        else:
            env = FullyObsWrapper(env, egocentric=False)
        
        env = RGBImgObsWrapper(env, tile_size=8, obs_size=84)
        env = FlattenObsWrapper(env)
        
        # Evaluate on test environment
        metrics = evaluate_model(model, env, agent_type, n_episodes)
        
        results.append({
            'seed': seed,
            'success_rate': metrics['success_rate'],
            'avg_return': metrics['avg_return'],
            'std_return': metrics['std_return'],
            'avg_steps': metrics['avg_steps']
        })
        
        print(f"  Success: {metrics['success_rate']*100:.1f}%, "
              f"Return: {metrics['avg_return']:.3f}, "
              f"Steps: {metrics['avg_steps']:.1f}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_env', default='MiniGrid-Empty-5x5-v0',
                       help='Environment model was trained on')
    parser.add_argument('--test_env', default='MiniGrid-Empty-8x8-v0',
                       help='Environment to test on')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', default='results/transfer_5x5_to_8x8.csv')
    
    args = parser.parse_args()
    
    # Test all configurations
    configs = [
        ('ppo', 'egocentric'),
        ('ppo', 'allocentric'),
        ('dqn', 'egocentric'),
        ('dqn', 'allocentric')
    ]
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"TRANSFER LEARNING EVALUATION")
    print(f"Train: {args.train_env} → Test: {args.test_env}")
    print(f"{'='*80}\n")
    
    for agent_type, obs_type in configs:
        results = test_transfer(
            args.train_env, 
            args.test_env, 
            agent_type, 
            obs_type, 
            args.seeds,
            args.n_episodes
        )
        
        if results:
            # Calculate aggregate statistics
            success_rates = [r['success_rate'] for r in results]
            mean_success = np.mean(success_rates)
            std_success = np.std(success_rates)
            
            all_results.append({
                'Train Env': args.train_env,
                'Test Env': args.test_env,
                'Algorithm': agent_type.upper(),
                'Observation': obs_type.capitalize(),
                'Mean Success (%)': f"{mean_success*100:.1f}",
                'Std Success (%)': f"{std_success*100:.1f}",
                'Seeds': results
            })
    
    # Create summary table
    if all_results:
        summary_rows = []
        for result in all_results:
            summary_rows.append({
                'Algorithm': result['Algorithm'],
                'Observation': result['Observation'],
                'Mean Success (%)': result['Mean Success (%)'],
                'Std Success (%)': result['Std Success (%)']
            })
        
        df = pd.DataFrame(summary_rows)
        
        print(f"\n{'='*80}")
        print("TRANSFER LEARNING RESULTS")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}\n")


if __name__ == '__main__':
    main()