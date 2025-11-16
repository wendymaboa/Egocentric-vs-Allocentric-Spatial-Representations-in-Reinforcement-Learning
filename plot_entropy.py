import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_tensorboard_data(log_dir, tags=['entropy', 'train/entropy', 'rollout/entropy', 'train/entropy_loss']):
    """
    Load data from TensorBoard event files
    
    Args:
        log_dir: Path to directory containing event files
        tags: List of tag names to search for entropy data
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    # Get available tags
    available_tags = ea.Tags()['scalars']
    
    # Find entropy tag
    entropy_tag = None
    for tag in tags:
        if tag in available_tags:
            entropy_tag = tag
            break
    
    if entropy_tag is None:
        # Try to find any tag with 'entropy' in it
        for tag in available_tags:
            if 'entropy' in tag.lower():
                entropy_tag = tag
                break
    
    if entropy_tag is None:
        return None, available_tags
    
    # Extract data
    events = ea.Scalars(entropy_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return {'steps': steps, 'values': values, 'tag': entropy_tag}, available_tags

def scan_tensorboard_logs(tensorboard_dir='tensorboard'):
    """
    Scan tensorboard directory structure and organize data by environment and algorithm
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    tensorboard_path = Path(tensorboard_dir)
    if not tensorboard_path.exists():
        print(f"Error: Directory '{tensorboard_dir}' not found!")
        return data
    
    print("=" * 80)
    print("Scanning TensorBoard logs...")
    print("=" * 80)
    
    # Scan environment folders
    for env_dir in tensorboard_path.iterdir():
        if not env_dir.is_dir():
            continue
        
        env_name = env_dir.name
        print(f"\n📁 Environment: {env_name}")
        
        # Scan algorithm folders (PPO_allocentric_1, PPO_egocentric_1, etc.)
        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            
            algo_name = algo_dir.name
            
            # Load tensorboard data
            entropy_data, available_tags = load_tensorboard_data(str(algo_dir))
            
            if entropy_data:
                data[env_name][algo_name] = entropy_data
                print(f"  ✓ {algo_name}: Found entropy data (tag: {entropy_data['tag']})")
            else:
                print(f"  ✗ {algo_name}: No entropy data found")
                if available_tags:
                    print(f"      Available tags: {available_tags[:5]}...")  # Show first 5 tags
    
    return data

def identify_observation_type(algo_name):
    """
    Identify observation type from algorithm folder name
    
    Examples:
        - PPO_allocentric_1 -> allocentric
        - PPO_egocentric_1 -> egocentric
        - DQN_allocentric_1 -> allocentric
    """
    algo_lower = algo_name.lower()
    
    if 'allocentric' in algo_lower:
        return 'allocentric'
    elif 'egocentric' in algo_lower:
        return 'egocentric'
    else:
        return 'unknown'

def get_algorithm_type(algo_name):
    """
    Extract algorithm type (PPO or DQN) from folder name
    """
    algo_upper = algo_name.upper()
    
    if 'PPO' in algo_upper:
        return 'PPO'
    elif 'DQN' in algo_upper:
        return 'DQN'
    else:
        return 'Unknown'

def plot_entropy_comparison(env_name, algo_data, output_dir='plots'):
    """
    Plot entropy comparison for an environment
    Shows separate lines for each algorithm + observation type combination
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define color scheme:
    # PPO-Allocentric: Blue
    # PPO-Egocentric: Red
    # DQN-Allocentric: Green
    # DQN-Egocentric: Orange
    
    color_map = {
        ('PPO', 'allocentric'): '#2E86AB',  # Blue
        ('PPO', 'egocentric'): '#E63946',   # Red
        ('DQN', 'allocentric'): '#06A77D',  # Green
        ('DQN', 'egocentric'): '#F77F00',   # Orange
        ('Unknown', 'unknown'): '#6C757D',  # Gray
    }
    
    # Organize data by algorithm type and observation type
    plotted_combos = set()
    
    for algo_name, data in sorted(algo_data.items()):
        algo_type = get_algorithm_type(algo_name)
        obs_type = identify_observation_type(algo_name)
        
        combo = (algo_type, obs_type)
        color = color_map.get(combo, color_map[('Unknown', 'unknown')])
        
        # Create label
        if obs_type != 'unknown':
            label = f"{algo_type} - {obs_type.capitalize()}"
        else:
            label = algo_name
        
        # Plot line
        ax.plot(data['steps'], data['values'], 
                linewidth=2.5, label=label, 
                color=color,
                alpha=0.85)
        
        plotted_combos.add(combo)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Policy Entropy', fontsize=14, fontweight='bold')
    ax.set_title(f'Policy Entropy Comparison - {env_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'entropy_{env_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: {output_file}")
    
    plt.show()

def plot_all_environments_grid(data, output_dir='plots'):
    """
    Create a grid of subplots showing entropy for all environments
    """
    n_envs = len(data)
    if n_envs == 0:
        print("No data to plot!")
        return
    
    # Calculate grid dimensions
    n_cols = min(2, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * n_cols, 8 * n_rows))
    if n_envs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    color_map = {
        ('PPO', 'allocentric'): '#2E86AB',  # Blue
        ('PPO', 'egocentric'): '#E63946',   # Red
        ('DQN', 'allocentric'): '#06A77D',  # Green
        ('DQN', 'egocentric'): '#F77F00',   # Orange
        ('Unknown', 'unknown'): '#6C757D',  # Gray
    }
    
    for idx, (env_name, algo_data) in enumerate(sorted(data.items())):
        ax = axes[idx]
        
        for algo_name, entropy_data in sorted(algo_data.items()):
            algo_type = get_algorithm_type(algo_name)
            obs_type = identify_observation_type(algo_name)
            
            combo = (algo_type, obs_type)
            color = color_map.get(combo, color_map[('Unknown', 'unknown')])
            
            # Create short label for grid
            if obs_type != 'unknown':
                label = f"{algo_type}-{obs_type[:4]}"
            else:
                label = algo_name
            
            ax.plot(entropy_data['steps'], entropy_data['values'],
                   linewidth=2, label=label,
                   color=color,
                   alpha=0.85)
        
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel('Entropy', fontsize=11)
        ax.set_title(env_name.replace('MiniGrid-', ''), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_envs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'entropy_all_environments_grid.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved: {output_file}")
    
    plt.show()

def plot_by_observation_type(data, output_dir='plots'):
    """
    Create separate plots comparing PPO vs DQN for each observation type
    """
    # Organize data by observation type
    by_obs_type = defaultdict(lambda: defaultdict(list))
    
    for env_name, algo_data in data.items():
        for algo_name, entropy_data in algo_data.items():
            obs_type = identify_observation_type(algo_name)
            algo_type = get_algorithm_type(algo_name)
            
            if obs_type != 'unknown':
                by_obs_type[obs_type][env_name].append({
                    'algo': algo_type,
                    'data': entropy_data,
                    'name': algo_name
                })
    
    # Plot for each observation type
    for obs_type in ['allocentric', 'egocentric']:
        if obs_type not in by_obs_type:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors_ppo = '#2E86AB' if obs_type == 'allocentric' else '#E63946'
        colors_dqn = '#06A77D' if obs_type == 'allocentric' else '#F77F00'
        
        for env_name, runs in sorted(by_obs_type[obs_type].items()):
            for run in runs:
                color = colors_ppo if run['algo'] == 'PPO' else colors_dqn
                label = f"{run['algo']} ({env_name.replace('MiniGrid-', '')})"
                
                ax.plot(run['data']['steps'], run['data']['values'],
                       linewidth=2.5, label=label, color=color, alpha=0.7)
        
        ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Policy Entropy', fontsize=14, fontweight='bold')
        ax.set_title(f'Policy Entropy - {obs_type.capitalize()} Observation', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f'entropy_{obs_type}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {output_file}")
        
        plt.show()

def main():
    # === CONFIGURATION ===
    tensorboard_dir = 'tensorboard'
    output_dir = 'plots'
    
    print("\n" + "="*80)
    print(" ENTROPY ANALYSIS - AUTO-DETECTION FROM FOLDER NAMES")
    print("="*80)
    
    # Load all tensorboard data
    data = scan_tensorboard_logs(tensorboard_dir)
    
    if not data:
        print("\n❌ No entropy data found in tensorboard logs!")
        print("\nTroubleshooting:")
        print("1. Check that 'tensorboard' folder exists")
        print("2. Verify event files are present")
        print("3. Ensure entropy was logged during training (PPO only)")
        print("4. Check folder names contain 'allocentric' or 'egocentric'")
        return
    
    print("\n" + "=" * 80)
    print(f"✓ Successfully loaded data from {len(data)} environment(s)")
    print("=" * 80)
    
    # Verify observation types detected
    print("\n" + "=" * 80)
    print("Detected observation types:")
    print("=" * 80)
    for env_name, algo_data in data.items():
        print(f"\n{env_name}:")
        for algo_name in sorted(algo_data.keys()):
            obs_type = identify_observation_type(algo_name)
            algo_type = get_algorithm_type(algo_name)
            print(f"  {algo_name} → {algo_type} + {obs_type}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    # Individual plots for each environment
    for env_name, algo_data in data.items():
        print(f"\n📊 Plotting {env_name}...")
        plot_entropy_comparison(env_name, algo_data, output_dir)
    
    # Combined grid view
    print("\n📊 Generating combined grid view...")
    plot_all_environments_grid(data, output_dir)
    
    # Plots organized by observation type
    print("\n📊 Generating observation type comparison plots...")
    plot_by_observation_type(data, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plots generated successfully!")
    print(f"✓ Plots saved to: {output_dir}/")
    print("=" * 80)
    
    print("\n📁 Generated files:")
    for file in sorted(Path(output_dir).glob('entropy_*.png')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()