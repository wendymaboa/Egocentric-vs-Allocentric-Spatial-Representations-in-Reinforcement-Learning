"""
Training with Stable-Baselines3 - Multiple Random Seeds Support
"""
import numpy as np
import os
import json
import argparse
import gym
import gym_minigrid
from datetime import datetime
import torch

# Your existing wrappers
import envs.envs
from envs.wrappers import *

# Stable-Baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


class FlattenObsWrapper(gym.ObservationWrapper):
    """Extract only image from dict observation for SB3"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
    
    def observation(self, obs):
        return obs['image']
    

class GymnasiumWrapper(gym.Wrapper):
    """Convert gym spaces to gymnasium spaces for SB3 compatibility"""
    def __init__(self, env):
        super().__init__(env)
        import gymnasium
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = gymnasium.spaces.Discrete(env.action_space.n)
        
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gymnasium.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype
            )


class Gym21CompatibilityWrapper(gym.Wrapper):
    """Make old gym environments compatible with new gym/gymnasium API"""
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs = self.env.reset(**kwargs)
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
    

class LoggingCallback(BaseCallback):
    """Custom callback for logging metrics with entropy tracking for PPO"""
    
    def __init__(self, log_dir, env_key, agent_type, observation_type, seed, eval_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.agent_type = agent_type
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{env_key}_{agent_type}_{observation_type}_seed{seed}_{timestamp}"
        self.log_file = os.path.join(log_dir, f"{self.run_name}.json")
        
        self.metrics = {
            'config': {
                'env_key': env_key,
                'agent_type': agent_type,
                'observation_type': observation_type,
                'seed': seed,
                'timestamp': timestamp
            },
            'episodes': [],
            'steps': [],
            'returns': [],
            'successes': [],
            'eval_returns': [],
            'eval_success_rates': [],
            'eval_episodes': [],
            'entropies': []  # For PPO only
        }
        
        self.episode_count = 0
    
    def _on_step(self):
        # Log entropy for PPO
        if self.agent_type == 'ppo' and hasattr(self.model, 'policy'):
            # Get current entropy from PPO's policy
            if hasattr(self.model.policy, 'get_distribution'):
                obs_tensor = torch.as_tensor(self.locals['obs_tensor']).to(self.model.device)
                with torch.no_grad():
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    entropy = distribution.entropy().mean().item()
                    self.metrics['entropies'].append(entropy)
        
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            
            # Get episode info
            info = self.locals['infos'][0]
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                
                # Log episode
                self.metrics['episodes'].append(self.episode_count)
                self.metrics['steps'].append(self.num_timesteps)
                self.metrics['returns'].append(ep_reward)
                self.metrics['successes'].append(1.0 if ep_reward > 0 else 0.0)
                
                if self.episode_count % 100 == 0:
                    recent_returns = self.metrics['returns'][-100:]
                    recent_success = self.metrics['successes'][-100:]
                    
                    avg_return = np.mean(recent_returns)
                    success_rate = np.mean(recent_success)
                    
                    # Calculate average entropy for PPO
                    entropy_str = ""
                    if self.agent_type == 'ppo' and self.metrics['entropies']:
                        recent_entropy = self.metrics['entropies'][-1000:]  # Last 1000 steps
                        avg_entropy = np.mean(recent_entropy)
                        entropy_str = f" | Avg Entropy: {avg_entropy:.3f}"
                    
                    print(f"Episode {self.episode_count} | Steps {self.num_timesteps} | "
                          f"Avg Return: {avg_return:.2f} | Success Rate: {success_rate:.2%}{entropy_str}")
                    
                    # Log evaluation metrics
                    self.metrics['eval_episodes'].append(self.episode_count)
                    self.metrics['eval_returns'].append(avg_return)
                    self.metrics['eval_success_rates'].append(success_rate)
                    
                    # Save periodically
                    self.save_metrics()
        
        return True
    
    def save_metrics(self):
        """Save metrics to JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _on_training_end(self):
        """Save final metrics"""
        self.save_metrics()
        print(f"\nTraining complete. Metrics saved to {self.log_file}")


# Environment configs
ENV_CONFIGS = {
    "MiniGrid-Empty-5x5-v0": {
        "total_timesteps": 50000,
        "learning_rate": 3e-4,
    },
    "MiniGrid-Empty-8x8-v0": {
        "total_timesteps": 100000,
        "learning_rate": 3e-4,
    },
    "MiniGrid-DoorKey-5x5-v0": {
        "total_timesteps": 500000,
        "learning_rate": 3e-4,
    },
    "MiniGrid-DoorKey-8x8-v0": {
        "total_timesteps": 1000000,
        "learning_rate": 3e-4,
    },
    "Minigrid-PickUpObj-Custom-v0": {
        "total_timesteps": 200000,
        "learning_rate": 3e-4,
    },
    "MiniGrid-GoToObject-6x6-N2-v0": {
        "total_timesteps": 200000,
        "learning_rate": 3e-4,
    },
    "MiniGrid-GoToObject-8x8-N2-v0": {
        "total_timesteps": 300000,
        "learning_rate": 3e-4,
    },
}


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make torch deterministic (can reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_env(env_key, egocentric=False, view_size=None, obs_size=84, seed=None):
    """Create environment with wrappers"""
    if "Custom" in env_key:
        env = gym.make(env_key)
    else:
        env = gym.make(env_key)
    
    # Set seed
    if seed is not None:
        env.seed(seed)
    
    # Apply observation wrappers
    if view_size is not None:
        observation_type = f"partial_egocentric_v{view_size}" if egocentric else f"partial_allocentric_v{view_size}"
        env = PartialObsWrapper(env, view_size=view_size, egocentric=egocentric)
    else:
        observation_type = "egocentric" if egocentric else "allocentric"
        env = FullyObsWrapper(env, egocentric=egocentric)
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=obs_size)
    env = FlattenObsWrapper(env)
    env = GymnasiumWrapper(env)
    env = Gym21CompatibilityWrapper(env)
    env = Monitor(env)
    
    return env, observation_type


def train_sb3(env_key, agent_type='ppo', egocentric=False, view_size=None, 
              obs_size=84, seed=0, save_model=True, load_model=False):
    """Train agent using Stable-Baselines3 with specified seed"""
    
    print(f"\n{'='*60}")
    print(f"Training with Stable-Baselines3 (Seed: {seed})")
    print(f"Environment: {env_key}")
    print(f"Agent: {agent_type.upper()}")
    print(f"{'='*60}\n")
    
    # Set random seed
    set_seed(seed)
    
    # Create environment
    env, observation_type = create_env(env_key, egocentric, view_size, obs_size, seed)
    env = DummyVecEnv([lambda: env])
    
    # Get config
    config = ENV_CONFIGS.get(env_key, {
        "total_timesteps": 100000,
        "learning_rate": 3e-4,
    })
    
    # Create directories
    model_dir = 'models'
    log_dir = 'logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Model path with seed
    model_path = os.path.join(model_dir, f'{env_key}_{agent_type}_{observation_type}_seed{seed}')
    
    # Create or load model
    if load_model and os.path.exists(model_path + '.zip'):
        print(f"Loading model from {model_path}.zip")
        if agent_type == 'ppo':
            model = PPO.load(model_path, env=env)
        else:
            model = DQN.load(model_path, env=env)
    else:
        print("Creating new model")
        
        tb_log_name = f"{agent_type.upper()}_{observation_type}_seed{seed}"
        tb_log_path = f"./tensorboard/{env_key}"
        
        if agent_type == 'ppo':
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=config['learning_rate'],
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=tb_log_path,
                seed=seed
            )
        else:
            model = DQN(
                "CnnPolicy",
                env,
                learning_rate=1e-4,
                buffer_size=100000,
                learning_starts=10000,
                batch_size=32,
                gamma=0.95,
                target_update_interval=1000,
                exploration_fraction=0.5,
                exploration_final_eps=0.1,
                verbose=1,
                tensorboard_log=tb_log_path,
                seed=seed
            )
    
    # Create callback
    callback = LoggingCallback(
        log_dir=log_dir,
        env_key=env_key,
        agent_type=agent_type,
        observation_type=observation_type,
        seed=seed,
        eval_freq=100
    )
    
    # Train
    print(f"\nStarting training for {config['total_timesteps']} timesteps...")
    print(f"Expected time: ~{config['total_timesteps'] // 10000} minutes with GPU\n")
    
    try:
        eval_callback = EvalCallback(
            env, 
            best_model_save_path=f'./models/best_seed{seed}/',
            log_path='./logs/',
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            callback_after_eval=None
        )
        
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=[callback, eval_callback],
            tb_log_name=tb_log_name,
            progress_bar=True
        )
        
        if save_model:
            model.save(model_path)
            print(f"\nModel saved to {model_path}.zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if save_model:
            model.save(model_path)
            print(f"Model saved to {model_path}.zip")
        callback.save_metrics()
    
    return model, callback.metrics


def train_multiple_seeds(env_key, agent_type='ppo', egocentric=False, view_size=None,
                         obs_size=84, seeds=[0, 1, 2], save_model=True):
    """Train with multiple random seeds"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING WITH MULTIPLE SEEDS: {seeds}")
    print(f"Environment: {env_key}")
    print(f"Agent: {agent_type.upper()}")
    print(f"Observation: {'Egocentric' if egocentric else 'Allocentric'}")
    print(f"{'='*80}\n")
    
    all_metrics = []
    
    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# SEED {seed}/{seeds[-1]}")
        print(f"{'#'*80}\n")
        
        model, metrics = train_sb3(
            env_key=env_key,
            agent_type=agent_type,
            egocentric=egocentric,
            view_size=view_size,
            obs_size=obs_size,
            seed=seed,
            save_model=save_model,
            load_model=False
        )
        
        all_metrics.append(metrics)
        
        print(f"\nSeed {seed} complete!")
        print(f"Final success rate: {metrics['eval_success_rates'][-1]:.2%}")
        print(f"Final avg return: {metrics['eval_returns'][-1]:.2f}")
    
    print(f"\n{'='*80}")
    print(f"ALL SEEDS COMPLETE")
    print(f"{'='*80}\n")
    
    # Print summary across seeds
    final_success_rates = [m['eval_success_rates'][-1] for m in all_metrics]
    final_returns = [m['eval_returns'][-1] for m in all_metrics]
    
    print(f"Success Rates: {final_success_rates}")
    print(f"Mean: {np.mean(final_success_rates):.2%} ± {np.std(final_success_rates):.2%}")
    print(f"\nReturns: {[f'{r:.2f}' for r in final_returns]}")
    print(f"Mean: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}\n")
    
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_key', default="MiniGrid-Empty-5x5-v0")
    parser.add_argument('--agent_type', default="ppo", choices=["ppo", "dqn"])
    parser.add_argument('--egocentric', action='store_true', default=False)
    parser.add_argument('--view_size', type=int, default=None)
    parser.add_argument('--obs_size', type=int, default=84)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='Random seeds to use (default: 0 1 2)')
    parser.add_argument('--single_seed', type=int, default=None,
                       help='Train single seed instead of multiple')
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Training will be slower.")
    
    # Train with single seed or multiple seeds
    if args.single_seed is not None:
        train_sb3(
            env_key=args.env_key,
            agent_type=args.agent_type,
            egocentric=args.egocentric,
            view_size=args.view_size,
            obs_size=args.obs_size,
            seed=args.single_seed,
            save_model=args.save_model,
            load_model=args.load_model
        )
    else:
        train_multiple_seeds(
            env_key=args.env_key,
            agent_type=args.agent_type,
            egocentric=args.egocentric,
            view_size=args.view_size,
            obs_size=args.obs_size,
            seeds=args.seeds,
            save_model=args.save_model
        )