"""
Visualize Stable-Baselines3 trained agents with extended evaluation
"""
import torch
import numpy as np
import os
import argparse
import gym
import gym_minigrid
import json

import envs.envs
from envs.wrappers import *
from matplotlib import pyplot as plt
import imageio

from stable_baselines3 import PPO, DQN


class FlattenObsWrapper(gym.ObservationWrapper):
    """Extract only image from dict observation"""
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
    

def fig_image(fig):
    """Convert matplotlib figure to image array"""
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def evaluate_model(model, env, agent_type, n_episodes=100, verbose=True):
    """Evaluate model over n episodes and return detailed statistics"""
    
    episode_rewards = []
    episode_steps = []
    episode_successes = []
    
    if verbose:
        print(f"\nEvaluating over {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs = env.reset()
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < 200:
            if agent_type == 'dqn':
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _ = model.predict(obs_transposed, deterministic=False)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        episode_successes.append(1.0 if episode_reward > 0 else 0.0)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{n_episodes} episodes")
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'success_rate': np.mean(episode_successes),
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_successes': episode_successes
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS ({n_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"Mean Steps: {stats['mean_steps']:.1f} ± {stats['std_steps']:.1f}")
        print(f"{'='*60}\n")
    
    return stats


def save_evaluation_results(stats, model_path, output_dir='results/evaluations'):
    """Save evaluation statistics to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(model_path).replace('.zip', '')
    output_file = os.path.join(output_dir, f'{model_name}_eval.json')
    
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {
        'mean_reward': float(stats['mean_reward']),
        'std_reward': float(stats['std_reward']),
        'mean_steps': float(stats['mean_steps']),
        'std_steps': float(stats['std_steps']),
        'success_rate': float(stats['success_rate']),
        'episode_rewards': [float(r) for r in stats['episode_rewards']],
        'episode_steps': [int(s) for s in stats['episode_steps']],
        'episode_successes': [float(s) for s in stats['episode_successes']]
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"Evaluation results saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_key', default="MiniGrid-Empty-5x5-v0")
    parser.add_argument('--agent_type', default="ppo", choices=["ppo", "dqn"])
    parser.add_argument('--egocentric', action='store_true', default=False)
    parser.add_argument('--view_size', type=int, default=None)
    parser.add_argument('--obs_size', type=int, default=84)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--seed', type=int, default=None, help='Specific seed model to load')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes to visualize')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Run 100-episode evaluation')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of episodes for evaluation')
    parser.add_argument('--save_eval', action='store_true', default=False,
                       help='Save evaluation results to JSON')
    
    args = parser.parse_args()
    
    # Create environment
    if "Custom" in args.env_key:
        env = gym.make(args.env_key)
    else:
        env = gym.make(args.env_key)
    
    # Apply wrappers
    if args.view_size is not None:
        observation_type = f"partial_egocentric_v{args.view_size}" if args.egocentric else f"partial_allocentric_v{args.view_size}"
        env = PartialObsWrapper(env, view_size=args.view_size, egocentric=args.egocentric)
    else:
        observation_type = "egocentric" if args.egocentric else "allocentric"
        env = FullyObsWrapper(env, egocentric=args.egocentric)
    
    env = RGBImgObsWrapper(env, tile_size=8, obs_size=args.obs_size)
    env = FlattenObsWrapper(env)
    
    # Determine model path
    if args.model_path:
        path = args.model_path
    else:
        # If seed specified, load that seed's model
        if args.seed is not None:
            path = f'models/{args.env_key}_{args.agent_type}_{observation_type}_seed{args.seed}.zip'
        else:
            path = f'models/{args.env_key}_{args.agent_type}_{observation_type}.zip'
    
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        print("\nAvailable models:")
        if os.path.exists('models/'):
            for f in os.listdir('models/'):
                if f.endswith('.zip'):
                    print(f"  - {f}")
        exit(1)
    
    print(f"Loading model from: {path}")
    
    # Load model
    if args.agent_type == 'ppo':
        model = PPO.load(path)
    else:
        model = DQN.load(path)
    
    print("Model loaded successfully!")
    print(f"Environment: {args.env_key}")
    print(f"Agent: {args.agent_type.upper()}")
    print(f"Observation: {observation_type}")
    
    # Run evaluation if requested
    if args.evaluate:
        eval_stats = evaluate_model(
            model, env, args.agent_type, 
            n_episodes=args.eval_episodes, 
            verbose=True
        )
        
        if args.save_eval:
            save_evaluation_results(eval_stats, path)
        
        # If only evaluating (not visualizing), exit here
        if not args.save and args.episodes <= 0:
            exit(0)
    
    # Visualization
    if args.save:
        print(f"\nSaving video of {args.episodes} episodes...")
        images = []
    else:
        print(f"\nVisualizing {args.episodes} episodes...")
        from gym_minigrid.window import Window
        window = Window(args.env_key)
    
    success_count = 0
    total_reward = 0
    total_steps = 0
    
    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        
        while not done and step < 200:
            if args.agent_type == 'dqn':
                obs_transposed = np.transpose(obs, (2, 0, 1))
                action, _states = model.predict(obs_transposed, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Render
            if not args.save:
                img = env.render('rgb_array')
                window.show_img(img)
            else:
                # Create side-by-side visualization
                image_allocentric = env.render("rgb_array", highlight=True)
                image_egocentric = obs
                
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
                axs[0].set_title("Global View", fontsize=16)
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].imshow(image_allocentric)
                
                axs[1].set_title(f"Agent's View ({observation_type})", fontsize=16)
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                axs[1].imshow(image_egocentric)
                
                fig.suptitle(f"Episode {episode+1} | Step: {step} | Reward: {episode_reward:.1f}", 
                            fontsize=14, y=0.95)
                
                images.append(fig_image(fig))
            
            # Step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if not args.save and hasattr(window, 'closed') and window.closed:
                break
        
        total_reward += episode_reward
        total_steps += step
        
        if episode_reward > 0:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAILED"
        
        print(f"{status} | Steps: {step} | Reward: {episode_reward:.1f}")
        
        if not args.save and hasattr(window, 'closed') and window.closed:
            break
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"Average Reward: {total_reward/args.episodes:.2f}")
    print(f"Average Steps: {total_steps/args.episodes:.1f}")
    print(f"{'='*60}")
    
    # Save GIF
    if args.save and images:
        os.makedirs("images", exist_ok=True)
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        output_path = f"images/{args.env_key}_{args.agent_type}_{observation_type}{seed_suffix}_sb3.gif"
        imageio.mimsave(output_path, images, fps=args.fps, loop=0)
        print(f"\nGIF saved to: {output_path}")