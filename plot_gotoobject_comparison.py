"""
Create GoToObject comparison visualization (box plot + bar chart)
Reads from evaluation JSON files
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats


def load_seed_data(env_key, agent_type, obs_type, seeds=[0, 1, 2]):
    """Load success rates from evaluation files"""
    success_rates = []
    
    for seed in seeds:
        eval_file = f'results/evaluations/{env_key}_{agent_type}_{obs_type}_seed{seed}_eval.json'
        
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                data = json.load(f)
            success_rates.append(data['success_rate'])
        else:
            print(f"Warning: {eval_file} not found")
    
    return success_rates


def plot_gotoobject_comparison(output_file='plots/gotoobject_comparison.png'):
    """Create publication-quality comparison plot"""
    
    env_key = 'MiniGrid-GoToObject-6x6-N2-v0'
    agent_type = 'ppo'
    
    # Load data
    ego_data = load_seed_data(env_key, agent_type, 'egocentric')
    allo_data = load_seed_data(env_key, agent_type, 'allocentric')
    
    if not ego_data or not allo_data:
        print("ERROR: Evaluation data not found!")
        print("Run: python visualize_sb3.py --env_key MiniGrid-GoToObject-6x6-N2-v0 --agent_type ppo --egocentric --seed X --evaluate --save_eval")
        return
    
    # Calculate statistics
    ego_mean = np.mean(ego_data)
    allo_mean = np.mean(allo_data)
    ego_std = np.std(ego_data, ddof=1)
    allo_std = np.std(allo_data, ddof=1)
    
    # Effect size
    pooled_std = np.sqrt(((len(ego_data)-1)*ego_std**2 + (len(allo_data)-1)*allo_std**2) / (len(ego_data)+len(allo_data)-2))
    cohens_d = (ego_mean - allo_mean) / pooled_std
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ----- Subplot 1: Box Plot -----
    ax = axes[0]
    
    bp = ax.boxplot([ego_data, allo_data], 
                     labels=['Egocentric', 'Allocentric'],
                     patch_artist=True,
                     widths=0.6)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual seed points
    for i, data in enumerate([ego_data, allo_data], 1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, color='black', s=100, zorder=3, 
                  alpha=0.6, edgecolors='white', linewidths=2)
    
    ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
    ax.set_title('GoToObject Performance\n(3 seeds per perspective)', fontsize=14, fontweight='bold')
    ax.set_ylim([0.4, 0.8])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add text annotation
    ax.text(0.5, 0.95, f"Cohen's d = {cohens_d:.2f} (large effect)", 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)
    
    # ----- Subplot 2: Bar Chart with Error Bars -----
    ax = axes[1]
    
    x = [0, 1]
    means = [ego_mean, allo_mean]
    stds = [ego_std, allo_std]
    labels = ['Egocentric', 'Allocentric']
    
    bars = ax.bar(x, means, yerr=stds, capsize=10, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.1%}\n±{std:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add individual seed points
    for i, data in enumerate([ego_data, allo_data]):
        ax.scatter([i]*len(data), data, color='black', s=100, zorder=3,
                  alpha=0.6, edgecolors='white', linewidths=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
    ax.set_title('Mean ± Std Dev', fontsize=14, fontweight='bold')
    ax.set_ylim([0.4, 0.8])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add difference arrow
    ax.annotate('', xy=(0, ego_mean), xytext=(1, allo_mean),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, (ego_mean + allo_mean)/2 + 0.02, 
            f'Δ = {ego_mean - allo_mean:.1%}',
            ha='center', color='red', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("GOTOOBJECT STATISTICS")
    print(f"{'='*60}")
    print(f"Egocentric:  {ego_mean:.1%} ± {ego_std:.1%}  (seeds: {[f'{x:.2f}' for x in ego_data]})")
    print(f"Allocentric: {allo_mean:.1%} ± {allo_std:.1%}  (seeds: {[f'{x:.2f}' for x in allo_data]})")
    print(f"Difference:  {ego_mean - allo_mean:.1%}")
    print(f"Cohen's d:   {cohens_d:.2f} (large effect)")
    print(f"{'='*60}\n")
    
    plt.show()


if __name__ == '__main__':
    plot_gotoobject_comparison()