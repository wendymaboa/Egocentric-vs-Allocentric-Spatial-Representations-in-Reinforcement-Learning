"""
Statistical Analysis for Multi-Seed RL Experiments
Performs appropriate tests given small sample sizes (n=3 seeds)
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_evaluation_results(env_key, agent_type, obs_type, seeds=[0, 1, 2], metric='success_rate'):
    """
    Load results for a specific configuration across seeds
    
    Returns: list of values [seed0_value, seed1_value, seed2_value]
    """
    values = []
    
    for seed in seeds:
        result_file = f'results/evaluations/{env_key}_{agent_type}_{obs_type}_seed{seed}_eval.json'
        
        if os.path.exists(result_file):
            import json
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            if metric == 'success_rate':
                values.append(data['success_rate'])
            elif metric == 'mean_reward':
                values.append(data['mean_reward'])
            elif metric == 'mean_steps':
                values.append(data['mean_steps'])
            elif metric == 'episode_rewards':
                values.extend(data['episode_rewards'])  # All episodes
        else:
            print(f"Warning: {result_file} not found")
    
    return values


def effect_size_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size
    
    Interpretation:
    - Small effect: d = 0.2
    - Medium effect: d = 0.5
    - Large effect: d = 0.8
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    Useful when sample size is small (n=3)
    """
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return lower, upper


def perform_statistical_tests(group1, group2, group1_name, group2_name, 
                              alpha=0.05, use_all_episodes=False):
    """
    Perform appropriate statistical tests given small sample size
    
    Args:
        group1, group2: Lists of values (either 3 seeds or 300 episodes from 3 seeds)
        group1_name, group2_name: Labels for groups
        alpha: Significance level
        use_all_episodes: If True, treats data as independent episodes
    """
    
    results = {
        'comparison': f"{group1_name} vs {group2_name}",
        'n1': len(group1),
        'n2': len(group2),
        'mean1': np.mean(group1),
        'mean2': np.mean(group2),
        'std1': np.std(group1, ddof=1),
        'std2': np.std(group2, ddof=1),
    }
    
    # 1. Independent t-test (assumes normality - questionable with n=3)
    if len(group1) >= 2 and len(group2) >= 2:
        t_stat, p_value_ttest = ttest_ind(group1, group2)
        results['t_test_statistic'] = t_stat
        results['t_test_p_value'] = p_value_ttest
        results['t_test_significant'] = p_value_ttest < alpha
    
    # 2. Mann-Whitney U test (non-parametric, better for small n)
    if len(group1) >= 2 and len(group2) >= 2:
        u_stat, p_value_mann = mannwhitneyu(group1, group2, alternative='two-sided')
        results['mann_whitney_u'] = u_stat
        results['mann_whitney_p_value'] = p_value_mann
        results['mann_whitney_significant'] = p_value_mann < alpha
    
    # 3. Effect size (Cohen's d) - Important for practical significance
    if len(group1) >= 2 and len(group2) >= 2:
        cohens_d = effect_size_cohens_d(group1, group2)
        results['cohens_d'] = cohens_d
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        results['effect_size_interpretation'] = effect
    
    # 4. Bootstrap confidence intervals (robust to small n)
    ci1_lower, ci1_upper = bootstrap_confidence_interval(group1)
    ci2_lower, ci2_upper = bootstrap_confidence_interval(group2)
    
    results['ci1_lower'] = ci1_lower
    results['ci1_upper'] = ci1_upper
    results['ci2_lower'] = ci2_lower
    results['ci2_upper'] = ci2_upper
    
    # Check if confidence intervals overlap
    results['ci_overlap'] = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
    results['ci_significant'] = not results['ci_overlap']
    
    return results


def compare_perspectives(env_key, agent_type, metric='success_rate', n_episodes=100):
    """
    Compare egocentric vs allocentric for a specific environment and algorithm
    """
    print(f"\n{'='*80}")
    print(f"Statistical Analysis: {env_key} - {agent_type.upper()}")
    print(f"Metric: {metric}")
    print(f"{'='*80}\n")
    
    # Load data for both perspectives
    if metric in ['success_rate', 'mean_reward', 'mean_steps']:
        # Aggregate statistics per seed (n=3)
        ego_values = load_evaluation_results(env_key, agent_type, 'egocentric', metric=metric)
        allo_values = load_evaluation_results(env_key, agent_type, 'allocentric', metric=metric)
        use_episodes = False
    else:
        # All episodes across seeds (n=300 if 100 episodes × 3 seeds)
        ego_values = load_evaluation_results(env_key, agent_type, 'egocentric', metric='episode_rewards')
        allo_values = load_evaluation_results(env_key, agent_type, 'allocentric', metric='episode_rewards')
        use_episodes = True
    
    if not ego_values or not allo_values:
        print("ERROR: Data not found. Make sure evaluation files exist.")
        return None
    
    # Perform tests
    results = perform_statistical_tests(
        ego_values, allo_values,
        f"{agent_type.upper()}-Egocentric",
        f"{agent_type.upper()}-Allocentric",
        use_all_episodes=use_episodes
    )
    
    # Print results
    print(f"Sample sizes: n_ego={results['n1']}, n_allo={results['n2']}")
    print(f"\nDescriptive Statistics:")
    print(f"  Egocentric:  {results['mean1']:.3f} ± {results['std1']:.3f}")
    print(f"  Allocentric: {results['mean2']:.3f} ± {results['std2']:.3f}")
    print(f"  Difference:  {results['mean1'] - results['mean2']:.3f}")
    
    if 't_test_p_value' in results:
        print(f"\nIndependent t-test:")
        print(f"  t-statistic: {results['t_test_statistic']:.3f}")
        print(f"  p-value: {results['t_test_p_value']:.4f}")
        print(f"  Significant: {results['t_test_significant']} (α=0.05)")
    
    if 'mann_whitney_p_value' in results:
        print(f"\nMann-Whitney U test (recommended for n=3):")
        print(f"  U-statistic: {results['mann_whitney_u']:.3f}")
        print(f"  p-value: {results['mann_whitney_p_value']:.4f}")
        print(f"  Significant: {results['mann_whitney_significant']} (α=0.05)")
    
    if 'cohens_d' in results:
        print(f"\nEffect Size (Cohen's d):")
        print(f"  d = {results['cohens_d']:.3f}")
        print(f"  Interpretation: {results['effect_size_interpretation']}")
    
    print(f"\nBootstrap 95% Confidence Intervals:")
    print(f"  Egocentric:  [{results['ci1_lower']:.3f}, {results['ci1_upper']:.3f}]")
    print(f"  Allocentric: [{results['ci2_lower']:.3f}, {results['ci2_upper']:.3f}]")
    print(f"  CIs overlap: {results['ci_overlap']}")
    print(f"  Significant by CI: {results['ci_significant']}")
    
    # Overall interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print(f"{'='*80}")
    
    if 'mann_whitney_p_value' in results:
        if results['mann_whitney_significant']:
            print(f"✓ SIGNIFICANT DIFFERENCE detected (p={results['mann_whitney_p_value']:.4f})")
            print(f"  Effect size: {results['effect_size_interpretation']} (d={results['cohens_d']:.3f})")
        else:
            print(f"✗ NO significant difference (p={results['mann_whitney_p_value']:.4f})")
            if abs(results['cohens_d']) > 0.5:
                print(f"  However, effect size is {results['effect_size_interpretation']} (d={results['cohens_d']:.3f})")
                print(f"  Larger sample size may reveal significance.")
    
    if results['ci_overlap']:
        print(f"  Confidence intervals overlap → difference not reliable")
    else:
        print(f"  Confidence intervals don't overlap → difference likely real")
    
    print(f"{'='*80}\n")
    
    return results


def compare_all_conditions(output_file='results/statistical_analysis.csv'):
    """
    Run statistical comparisons for all environment × algorithm combinations
    """
    
    envs = [
        'MiniGrid-Empty-5x5-v0',
        'MiniGrid-Empty-8x8-v0',
        'MiniGrid-DoorKey-5x5-v0',
        'MiniGrid-GoToObject-6x6-N2-v0'
    ]
    
    agents = ['ppo', 'dqn']
    metric = 'success_rate'
    
    all_results = []
    
    for env in envs:
        for agent in agents:
            result = compare_perspectives(env, agent, metric)
            if result:
                all_results.append(result)
    
    # Create summary table
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Select key columns
        summary_cols = [
            'comparison', 'n1', 'n2',
            'mean1', 'std1', 'mean2', 'std2',
            'mann_whitney_p_value', 'mann_whitney_significant',
            'cohens_d', 'effect_size_interpretation',
            'ci_significant'
        ]
        
        df_summary = df[summary_cols]
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_summary.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(df_summary.to_string(index=False))
        print(f"{'='*80}\n")
        print(f"Results saved to: {output_file}")
        
        return df_summary
    
    return None


def visualize_comparison(env_key, agent_type, metric='success_rate', output_dir='plots/statistical'):
    """
    Create visualization of comparison with confidence intervals
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    ego_values = load_evaluation_results(env_key, agent_type, 'egocentric', metric=metric)
    allo_values = load_evaluation_results(env_key, agent_type, 'allocentric', metric=metric)
    
    if not ego_values or not allo_values:
        print(f"Cannot create visualization: data not found")
        return
    
    # Calculate statistics
    ego_mean = np.mean(ego_values)
    allo_mean = np.mean(allo_values)
    ego_std = np.std(ego_values, ddof=1)
    allo_std = np.std(allo_values, ddof=1)
    
    # Bootstrap CIs
    ego_ci = bootstrap_confidence_interval(ego_values)
    allo_ci = bootstrap_confidence_interval(allo_values)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Bar chart with error bars
    ax = axes[0]
    x = [0, 1]
    means = [ego_mean, allo_mean]
    stds = [ego_std, allo_std]
    labels = ['Egocentric', 'Allocentric']
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(x, means, yerr=stds, capsize=10, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{env_key} - {agent_type.upper()}\nMean ± Std', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add individual seed points
    for i, values in enumerate([ego_values, allo_values]):
        ax.scatter([i]*len(values), values, color='black', s=100, zorder=3, 
                   alpha=0.6, edgecolors='white', linewidths=1.5)
    
    # Subplot 2: Confidence intervals
    ax = axes[1]
    
    # Plot CIs as horizontal bars
    ax.barh([0], [ego_mean], xerr=[[ego_mean - ego_ci[0]], [ego_ci[1] - ego_mean]], 
            height=0.3, color=colors[0], alpha=0.7, label='Egocentric', capsize=5)
    ax.barh([1], [allo_mean], xerr=[[allo_mean - allo_ci[0]], [allo_ci[1] - allo_mean]], 
            height=0.3, color=colors[1], alpha=0.7, label='Allocentric', capsize=5)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Bootstrap 95% CI', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, f'{env_key}_{agent_type}_{metric}_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def variance_analysis(env_key, agent_type, obs_type, output_dir='plots/statistical'):
    """
    Analyze variance across seeds for a specific configuration
    Useful for detecting training instability
    """
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = [0, 1, 2]
    success_rates = []
    
    for seed in seeds:
        values = load_evaluation_results(env_key, agent_type, obs_type, seeds=[seed], 
                                        metric='success_rate')
        if values:
            success_rates.append(values[0])
    
    if len(success_rates) < 3:
        print(f"Not enough data for variance analysis")
        return
    
    # Calculate variance metrics
    mean_sr = np.mean(success_rates)
    std_sr = np.std(success_rates, ddof=1)
    cv = (std_sr / mean_sr) * 100 if mean_sr > 0 else 0  # Coefficient of variation
    
    print(f"\n{'='*80}")
    print(f"Variance Analysis: {env_key} - {agent_type.upper()} - {obs_type}")
    print(f"{'='*80}")
    print(f"Success rates across seeds: {success_rates}")
    print(f"Mean: {mean_sr:.3f}")
    print(f"Std Dev: {std_sr:.3f}")
    print(f"Coefficient of Variation: {cv:.1f}%")
    
    if cv > 30:
        print(f"⚠️  HIGH VARIANCE DETECTED (CV > 30%)")
        print(f"   Training is UNSTABLE for this configuration")
    elif cv > 15:
        print(f"⚠️  Moderate variance (CV = {cv:.1f}%)")
    else:
        print(f"✓  Low variance - stable training")
    
    print(f"{'='*80}\n")
    
    return {
        'env': env_key,
        'agent': agent_type,
        'obs': obs_type,
        'mean': mean_sr,
        'std': std_sr,
        'cv': cv,
        'success_rates': success_rates
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical Analysis for RL Experiments')
    parser.add_argument('--mode', choices=['compare', 'variance', 'all'], default='all',
                       help='Analysis mode')
    parser.add_argument('--env_key', default=None, 
                       help='Specific environment to analyze')
    parser.add_argument('--agent_type', default=None,
                       help='Specific agent type')
    parser.add_argument('--metric', default='success_rate',
                       choices=['success_rate', 'mean_reward', 'mean_steps'])
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS FOR RL EXPERIMENTS")
    print("="*80 + "\n")
    
    if args.mode in ['compare', 'all']:
        print("RUNNING PERSPECTIVE COMPARISONS...\n")
        
        if args.env_key and args.agent_type:
            # Single comparison
            result = compare_perspectives(args.env_key, args.agent_type, args.metric)
            
            if args.visualize:
                visualize_comparison(args.env_key, args.agent_type, args.metric)
        else:
            # All comparisons
            compare_all_conditions()
    
    if args.mode in ['variance', 'all']:
        print("\nRUNNING VARIANCE ANALYSIS...\n")
        
        envs = ['MiniGrid-Empty-5x5-v0', 'MiniGrid-Empty-8x8-v0', 
                'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-GoToObject-6x6-N2-v0']
        agents = ['ppo', 'dqn']
        obs_types = ['egocentric', 'allocentric']
        
        variance_results = []
        
        for env in envs:
            for agent in agents:
                for obs in obs_types:
                    result = variance_analysis(env, agent, obs)
                    if result:
                        variance_results.append(result)
        
        # Summary table
        if variance_results:
            df = pd.DataFrame(variance_results)
            print("\n" + "="*80)
            print("VARIANCE SUMMARY TABLE")
            print("="*80)
            print(df[['env', 'agent', 'obs', 'mean', 'std', 'cv']].to_string(index=False))
            print("="*80 + "\n")
            
            # Save
            df.to_csv('results/variance_analysis.csv', index=False)
            print("Saved: results/variance_analysis.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/statistical_analysis.csv")
    print("  - results/variance_analysis.csv")
    print("  - plots/statistical/*.png (if --visualize used)")