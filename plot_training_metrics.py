"""Plot Training Metrics with Multi-Seed Support (Mean ± Std) - FIXED & FLEXIBLE"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from collections import defaultdict

# Increase default font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def smooth_curve(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def load_log(log_file):
    """Load a single log file"""
    with open(log_file, 'r') as f:
        return json.load(f)

def group_logs_by_config(log_files):
    """Group log files by environment, agent, and observation type"""
    groups = defaultdict(list)
    
    for log_file in log_files:
        try:
            data = load_log(log_file)
            config = data['config']
            
            # Create key without seed
            key = (
                config['env_key'],
                config['agent_type'],
                config['observation_type']
            )
            
            groups[key].append(log_file)
        except Exception as e:
            print(f"Warning: Could not load {log_file}: {e}")
            continue
    
    return groups

def plot_multi_seed_comparison(log_files, output_dir='plots', use_timesteps=False, 
                               plot_types=['returns', 'success', 'timesteps', 'entropy']):
    """
    Plot mean ± std across multiple seeds with flexible subplot selection
    
    Args:
        plot_types: List of plots to include. Options:
            'returns' - Episode returns
            'success' - Success rate
            'timesteps' - Timesteps vs episodes
            'entropy' - Policy entropy (PPO only)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by config (excluding seed)
    groups = group_logs_by_config(log_files)
    
    print(f"\nFound {len(groups)} unique configurations:")
    for key in groups:
        env_key, agent_type, obs_type = key
        print(f"  {env_key} - {agent_type.upper()} - {obs_type}: {len(groups[key])} seeds")
    
    # Plot each group
    for key, group_files in groups.items():
        if len(group_files) < 1:
            print(f"\nSkipping {key}: no seeds found")
            continue
        
        env_key, agent_type, obs_type = key
        print(f"\nPlotting {env_key} - {agent_type.upper()} - {obs_type} ({len(group_files)} seeds)")
        
        # Load all data
        all_data = [load_log(f) for f in group_files]
        
        # Determine which plots to show
        available_plots = []
        if 'returns' in plot_types:
            available_plots.append(('returns', 'Episode Returns'))
        if 'success' in plot_types:
            available_plots.append(('success', 'Success Rate'))
        if 'timesteps' in plot_types:
            available_plots.append(('timesteps', 'Timesteps vs Episodes'))
        if 'entropy' in plot_types and agent_type == 'ppo':
            # Check if entropy data exists
            has_entropy = any(data.get('entropies') for data in all_data)
            if has_entropy:
                available_plots.append(('entropy', 'Policy Entropy'))
            else:
                print(f"  Warning: No entropy data found for PPO model")
        
        if not available_plots:
            print(f"  No plots to generate")
            continue
        
        # Determine grid size
        n_plots = len(available_plots)
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_plots == 3:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        
        fig.suptitle(f'{env_key} - {agent_type.upper()} - {obs_type}\n({len(group_files)} seeds)', 
                     fontsize=20, fontweight='bold')
        
        x_label = 'Timesteps' if use_timesteps else 'Episode'
        
        # Plot each selected type
        for idx, (plot_type, title) in enumerate(available_plots):
            ax = axes[idx]
            
            if plot_type == 'returns':
                plot_metric_with_std(ax, all_data, 'returns', x_label, 'Return', 
                                    'Episode Returns', use_timesteps, smooth_window=50)
            elif plot_type == 'success':
                plot_success_rate_with_std(ax, all_data, x_label, use_timesteps, window=100)
            elif plot_type == 'timesteps':
                plot_timesteps_episodes(ax, all_data, use_timesteps)
            elif plot_type == 'entropy':
                plot_entropy_with_std(ax, all_data)
        
        # Hide unused subplots
        for idx in range(len(available_plots), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        suffix = '_timesteps' if use_timesteps else '_episodes'
        plot_suffix = '_' + '_'.join(plot_types)
        env_short = env_key.replace('MiniGrid-', '').replace('-v0', '')
        output_file = os.path.join(output_dir, 
                                   f'{env_short}_{agent_type}_{obs_type}_multiseed{plot_suffix}{suffix}.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

def plot_metric_with_std(ax, all_data, metric_key, x_label, y_label, title, 
                         use_timesteps, smooth_window=50):
    """Plot metric with mean ± std shading"""
    
    # Get all runs' data
    all_runs = []
    for data in all_data:
        if not data.get(metric_key):
            continue
        
        x_data = data.get('steps', []) if use_timesteps else data.get('episodes', [])
        y_data = data.get(metric_key, [])
        
        if not x_data or not y_data:
            continue
        
        # Smooth if long enough
        if len(y_data) > smooth_window:
            y_data = smooth_curve(y_data, window=smooth_window)
            x_data = x_data[smooth_window-1:]
        
        all_runs.append((x_data, y_data))
    
    if not all_runs:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        return
    
    # Find common x range
    max_x = min(max(x) for x, y in all_runs)
    
    # Interpolate all runs to common x grid
    x_common = np.linspace(0, max_x, 500)
    y_interpolated = []
    
    for x_data, y_data in all_runs:
        y_interp = np.interp(x_common, x_data, y_data)
        y_interpolated.append(y_interp)
    
    # Calculate mean and std
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    
    # Plot with larger line width
    ax.plot(x_common, y_mean, linewidth=3, label='Mean', color='#2E86AB')
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, 
                     alpha=0.3, label='±1 Std', color='#2E86AB')
    
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=13)

def plot_success_rate_with_std(ax, all_data, x_label, use_timesteps, window=100):
    """Plot success rate with mean ± std"""
    
    all_runs = []
    for data in all_data:
        if not data.get('successes'):
            continue
        
        x_data = data.get('steps', []) if use_timesteps else data.get('episodes', [])
        successes = data.get('successes', [])
        
        if not x_data or not successes:
            continue
        
        # Calculate rolling success rate
        if len(successes) >= window:
            success_rate = []
            x_windowed = []
            for i in range(len(successes) - window + 1):
                success_rate.append(np.mean(successes[i:i+window]))
                x_windowed.append(x_data[i + window - 1])
            all_runs.append((np.array(x_windowed), np.array(success_rate)))
    
    if not all_runs:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        return
    
    # Find common x range
    max_x = min(max(x) for x, y in all_runs)
    
    # Interpolate
    x_common = np.linspace(0, max_x, 500)
    y_interpolated = []
    
    for x_data, y_data in all_runs:
        y_interp = np.interp(x_common, x_data, y_data)
        y_interpolated.append(y_interp)
    
    # Calculate mean and std
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    
    # Plot with larger line width
    ax.plot(x_common, y_mean, linewidth=3, label='Mean', color='#A23B72')
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, 
                     alpha=0.3, label='±1 Std', color='#A23B72')
    
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=15, fontweight='bold')
    ax.set_title(f'Success Rate (Rolling {window} episodes)', fontsize=16, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=13)

def plot_timesteps_episodes(ax, all_data, use_timesteps):
    """Plot timesteps vs episodes for all seeds"""
    
    has_data = False
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_data)))
    
    for i, data in enumerate(all_data):
        if not data.get('steps') or not data.get('episodes'):
            continue
        
        has_data = True
        seed = data['config'].get('seed', i)
        
        if use_timesteps:
            ax.plot(data['steps'], data['episodes'], alpha=0.7, linewidth=2.5, 
                   label=f'Seed {seed}', color=colors[i])
            ax.set_xlabel('Timesteps', fontsize=15, fontweight='bold')
            ax.set_ylabel('Episode', fontsize=15, fontweight='bold')
            ax.set_title('Episode vs Timesteps', fontsize=16, fontweight='bold')
        else:
            ax.plot(data['episodes'], data['steps'], alpha=0.7, linewidth=2.5,
                   label=f'Seed {seed}', color=colors[i])
            ax.set_xlabel('Episode', fontsize=15, fontweight='bold')
            ax.set_ylabel('Total Timesteps', fontsize=15, fontweight='bold')
            ax.set_title('Timesteps vs Episodes', fontsize=16, fontweight='bold')
    
    if has_data:
        ax.legend(fontsize=13, loc='best')
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=13)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)

def plot_entropy_with_std(ax, all_data):
    """Plot entropy with mean ± std (PPO only) - FIXED"""
    
    all_runs = []
    max_len = 0
    
    for data in all_data:
        entropies = data.get('entropies', [])
        
        if not entropies or len(entropies) == 0:
            print(f"  Warning: No entropy data in this seed")
            continue
        
        # Don't smooth if data is too short
        if len(entropies) > 100:
            smoothed = smooth_curve(entropies, window=100)
            x = np.arange(len(smoothed))
            all_runs.append((x, smoothed))
            max_len = max(max_len, len(smoothed))
        else:
            x = np.arange(len(entropies))
            all_runs.append((x, np.array(entropies)))
            max_len = max(max_len, len(entropies))
    
    if not all_runs:
        ax.text(0.5, 0.5, 'No entropy data\n(Check if entropy was logged during training)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Find common length
    min_len = min(len(y) for x, y in all_runs)
    
    if min_len == 0:
        ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        return
    
    # Trim to common length
    y_all = [y[:min_len] for x, y in all_runs]
    x_common = np.arange(min_len)
    
    # Calculate mean and std
    y_mean = np.mean(y_all, axis=0)
    y_std = np.std(y_all, axis=0)
    
    # Plot with larger line width
    ax.plot(x_common, y_mean, linewidth=3, label='Mean', color='#F18F01')
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, 
                     alpha=0.3, label='±1 Std', color='#F18F01')
    
    ax.set_xlabel('Training Step', fontsize=15, fontweight='bold')
    ax.set_ylabel('Entropy', fontsize=15, fontweight='bold')
    ax.set_title('Policy Entropy (PPO)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.tick_params(labelsize=13)

def plot_cross_config_comparison(log_files, comparison_type='observation', 
                                 output_dir='plots', use_timesteps=False,
                                 plot_types=['returns', 'success', 'timesteps']):
    """Compare different configs (e.g., ego vs allo) with multi-seed support"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by config
    groups = group_logs_by_config(log_files)
    
    # Further group by environment for comparison
    env_groups = defaultdict(lambda: defaultdict(list))
    
    for key, files in groups.items():
        env_key, agent_type, obs_type = key
        env_groups[env_key][(agent_type, obs_type)] = files
    
    # Plot comparisons for each environment
    for env_key, configs in env_groups.items():
        if len(configs) < 2:
            continue
        
        print(f"\nPlotting comparison for {env_key}")
        
        # Determine subplot layout
        n_plots = len(plot_types)
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_plots == 3:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        
        if comparison_type == 'observation':
            fig.suptitle(f'{env_key} - Comparison: Egocentric vs Allocentric', 
                        fontsize=20, fontweight='bold')
        elif comparison_type == 'agent':
            fig.suptitle(f'{env_key} - Comparison: DQN vs PPO', 
                        fontsize=20, fontweight='bold')
        else:
            fig.suptitle(f'{env_key} - Training Comparison', 
                        fontsize=20, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        x_label = 'Timesteps' if use_timesteps else 'Episode'
        
        # Plot each metric
        for idx, plot_type in enumerate(plot_types):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            for config_idx, ((agent_type, obs_type), files) in enumerate(configs.items()):
                color = colors[config_idx]
                label = f"{agent_type.upper()}-{obs_type}"
                
                # Load all seeds for this config
                all_data = [load_log(f) for f in files]
                
                if plot_type == 'returns':
                    plot_comparison_metric(ax, all_data, 'returns', x_label, 'Return',
                                          'Episode Returns', use_timesteps, label, color, smooth_window=50)
                elif plot_type == 'success':
                    plot_comparison_success(ax, all_data, x_label, use_timesteps, label, color, window=100)
                elif plot_type == 'timesteps':
                    plot_comparison_timesteps(ax, all_data, use_timesteps, label, color)
        
        # Add legends and formatting
        for idx in range(min(len(plot_types), len(axes))):
            ax = axes[idx]
            if ax.lines:
                ax.legend(fontsize=13, loc='best')
            ax.grid(True, alpha=0.3, linewidth=1.5)
            ax.tick_params(labelsize=13)
        
        # Hide unused subplots
        for idx in range(len(plot_types), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save
        suffix = '_timesteps' if use_timesteps else '_episodes'
        plot_suffix = '_' + '_'.join(plot_types)
        env_short = env_key.replace('MiniGrid-', '').replace('-v0', '')
        
        if comparison_type == 'observation':
            output_file = os.path.join(output_dir, f'comparison_{env_short}_ego_vs_allo{plot_suffix}{suffix}.png')
        elif comparison_type == 'agent':
            output_file = os.path.join(output_dir, f'comparison_{env_short}_dqn_vs_ppo{plot_suffix}{suffix}.png')
        else:
            output_file = os.path.join(output_dir, f'comparison_{env_short}_custom{plot_suffix}{suffix}.png')
        
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

def plot_comparison_metric(ax, all_data, metric_key, x_label, y_label, title, 
                          use_timesteps, label, color, smooth_window=50):
    """Plot single metric for comparison with mean ± std"""
    
    all_runs = []
    for data in all_data:
        if not data.get(metric_key):
            continue
        
        x_data = data.get('steps', []) if use_timesteps else data.get('episodes', [])
        y_data = data.get(metric_key, [])
        
        if not x_data or not y_data:
            continue
        
        if len(y_data) > smooth_window:
            y_data = smooth_curve(y_data, window=smooth_window)
            x_data = x_data[smooth_window-1:]
        
        all_runs.append((x_data, y_data))
    
    if not all_runs:
        return
    
    # Find common x range
    max_x = min(max(x) for x, y in all_runs)
    
    # Interpolate
    x_common = np.linspace(0, max_x, 500)
    y_interpolated = []
    
    for x_data, y_data in all_runs:
        y_interp = np.interp(x_common, x_data, y_data)
        y_interpolated.append(y_interp)
    
    # Calculate mean and std
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    
    # Plot
    ax.plot(x_common, y_mean, linewidth=3, label=label, color=color)
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
    
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=15, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

def plot_comparison_success(ax, all_data, x_label, use_timesteps, label, color, window=100):
    """Plot success rate for comparison with mean ± std"""
    
    all_runs = []
    for data in all_data:
        if not data.get('successes'):
            continue
        
        x_data = data.get('steps', []) if use_timesteps else data.get('episodes', [])
        successes = data.get('successes', [])
        
        if not x_data or not successes:
            continue
        
        if len(successes) >= window:
            success_rate = []
            x_windowed = []
            for i in range(len(successes) - window + 1):
                success_rate.append(np.mean(successes[i:i+window]))
                x_windowed.append(x_data[i + window - 1])
            all_runs.append((np.array(x_windowed), np.array(success_rate)))
    
    if not all_runs:
        return
    
    # Find common x range
    max_x = min(max(x) for x, y in all_runs)
    
    # Interpolate
    x_common = np.linspace(0, max_x, 500)
    y_interpolated = []
    
    for x_data, y_data in all_runs:
        y_interp = np.interp(x_common, x_data, y_data)
        y_interpolated.append(y_interp)
    
    # Calculate mean and std
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    
    # Plot
    ax.plot(x_common, y_mean, linewidth=3, label=label, color=color)
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
    
    ax.set_xlabel(x_label, fontsize=15, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=15, fontweight='bold')
    ax.set_title(f'Success Rate (Rolling {window} episodes)', fontsize=16, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])

def plot_comparison_timesteps(ax, all_data, use_timesteps, label, color):
    """Plot timesteps vs episodes comparison with mean ± std"""
    
    all_runs = []
    for data in all_data:
        if not data.get('steps') or not data.get('episodes'):
            continue
        
        if use_timesteps:
            all_runs.append((data['steps'], data['episodes']))
        else:
            all_runs.append((data['episodes'], data['steps']))
    
    if not all_runs:
        return
    
    # Find common x range
    max_x = min(max(x) for x, y in all_runs)
    
    # Interpolate
    x_common = np.linspace(0, max_x, 500)
    y_interpolated = []
    
    for x_data, y_data in all_runs:
        y_interp = np.interp(x_common, x_data, y_data)
        y_interpolated.append(y_interp)
    
    # Calculate mean and std
    y_mean = np.mean(y_interpolated, axis=0)
    y_std = np.std(y_interpolated, axis=0)
    
    # Plot
    ax.plot(x_common, y_mean, linewidth=3, label=label, color=color)
    ax.fill_between(x_common, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
    
    if use_timesteps:
        ax.set_xlabel('Timesteps', fontsize=15, fontweight='bold')
        ax.set_ylabel('Episode', fontsize=15, fontweight='bold')
        ax.set_title('Episode vs Timesteps', fontsize=16, fontweight='bold')
    else:
        ax.set_xlabel('Episode', fontsize=15, fontweight='bold')
        ax.set_ylabel('Total Timesteps', fontsize=15, fontweight='bold')
        ax.set_title('Timesteps vs Episodes', fontsize=16, fontweight='bold')

def find_logs(pattern='logs/*.json'):
    """Find all log files matching pattern"""
    return glob.glob(pattern)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training metrics with multi-seed support')
    parser.add_argument('--log_dir', default='logs', help='Directory containing log files')
    parser.add_argument('--plot_dir', default='plots', help='Output directory for plots')
    parser.add_argument('--mode', choices=['multiseed', 'compare', 'all'], default='all',
                       help='multiseed: aggregate seeds, compare: compare configs, all: both')
    parser.add_argument('--plot_types', nargs='+', 
                       choices=['returns', 'success', 'timesteps', 'entropy'],
                       default=['returns', 'success', 'timesteps', 'entropy'],
                       help='Which plots to generate (space-separated list)')
    parser.add_argument('--log_files', nargs='+', help='Specific log files to plot')
    parser.add_argument('--env_key', default=None, help='Filter by environment')
    parser.add_argument('--agent_type', default=None, help='Filter by agent type')
    parser.add_argument('--comparison', choices=['observation', 'agent', 'custom'],
                       default='observation', help='Type of comparison')
    parser.add_argument('--use_timesteps', action='store_true',
                       help='Use timesteps instead of episodes on x-axis')
    
    args = parser.parse_args()
    
    # Find log files
    if args.log_files:
        log_files = args.log_files
    else:
        log_files = find_logs(f'{args.log_dir}/*.json')
    
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return
    
    # Filter if requested
    if args.env_key or args.agent_type:
        filtered = []
        for log_file in log_files:
            try:
                data = load_log(log_file)
                config = data['config']
                
                if args.env_key and config['env_key'] != args.env_key:
                    continue
                if args.agent_type and config['agent_type'] != args.agent_type:
                    continue
                
                filtered.append(log_file)
            except:
                continue
        log_files = filtered
    
    print(f"Found {len(log_files)} log files")
    print(f"Plots to generate: {', '.join(args.plot_types)}")
    if args.use_timesteps:
        print("Using timesteps on x-axis")
    
    # Plot based on mode
    if args.mode in ['multiseed', 'all']:
        print("\n" + "="*80)
        print("PLOTTING MULTI-SEED AGGREGATIONS (Mean ± Std)")
        print("="*80)
        plot_multi_seed_comparison(log_files, args.plot_dir, args.use_timesteps, args.plot_types)
    
    if args.mode in ['compare', 'all']:
        print("\n" + "="*80)
        print("PLOTTING CROSS-CONFIG COMPARISONS")
        print("="*80)
        # For comparison mode, exclude entropy by default (can cause issues with mixed PPO/DQN)
        compare_plot_types = [p for p in args.plot_types if p != 'entropy']
        plot_cross_config_comparison(log_files, args.comparison, args.plot_dir, 
                                     args.use_timesteps, compare_plot_types)
    
    print(f"\nAll plots saved to {args.plot_dir}/")

if __name__ == '__main__':
    main()