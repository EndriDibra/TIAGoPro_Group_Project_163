#!/usr/bin/env python3
"""
Generate plots and statistics from metrics CSV.

Usage:
    python3 plot_metrics.py --input all_metrics.csv --output-dir plots/

This script generates:
1. Bar charts comparing experiments by scenario
2. Box plots showing metric distributions
3. Stacked bar chart of VLM actions
4. Statistical summary with tests
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats


# Display name mappings
EXPERIMENT_NAMES = {
    'mistral': 'Cloud VLM',
    'smol': 'Local VLM',
    'novlm': 'No VLM',
    'notrack': 'No Tracking'
}

SCENARIO_NAMES = {
    'frontal_approach': 'Frontal Approach',
    'intersection': 'Intersection',
    'doorway': 'Doorway',
    'narrow_doorway': 'Doorway'
}

# Consistent color scheme
COLORS = {
    'Cloud VLM': '#e74c3c',    # Red
    'Local VLM': '#3498db',    # Blue
    'No VLM': '#2ecc71',       # Green
    'No Tracking': '#9b59b6'   # Purple
}


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Load CSV and preprocess data."""
    df = pd.read_csv(csv_path)
    
    # Handle 'inf' values in min_ttc
    df['min_ttc'] = df['min_ttc'].replace('inf', np.inf)
    df['min_ttc'] = pd.to_numeric(df['min_ttc'], errors='coerce')
    
    # Handle 'N/A' values in latency columns
    for col in ['detection_latency', 'vlm_latency']:
        if col in df.columns:
            df[col] = df[col].replace('N/A', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add collision indicator
    df['collision'] = df['min_distance'] < 0
    
    # Map to display names
    df['experiment_display'] = df['experiment'].map(EXPERIMENT_NAMES).fillna(df['experiment'])
    df['scenario_display'] = df['scenario'].map(SCENARIO_NAMES).fillna(df['scenario'])
    
    return df


def get_experiment_order(df: pd.DataFrame) -> list:
    """Get experiments in consistent order."""
    order = ['No VLM', 'Local VLM', 'Cloud VLM', 'No Tracking']
    available = df['experiment_display'].unique()
    return [e for e in order if e in available]


def get_colors_for_experiments(experiments: list) -> list:
    """Get colors matching experiment order."""
    return [COLORS.get(exp, '#95a5a6') for exp in experiments]


def plot_psc_by_experiment(df: pd.DataFrame, output_dir: str):
    """Bar chart: PSC by experiment and scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by experiment and scenario
    pivot = df.pivot_table(
        values='psc', 
        index='scenario_display', 
        columns='experiment_display', 
        aggfunc='mean'
    )
    
    # Reorder columns
    exp_order = get_experiment_order(df)
    pivot = pivot[[c for c in exp_order if c in pivot.columns]]
    
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black',
               color=get_colors_for_experiments(pivot.columns))
    
    ax.set_ylabel('Personal Space Compliance (%)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('PSC by Experiment and Scenario (Mean)', fontsize=14)
    ax.legend(title='Experiment', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psc_by_experiment.svg'))
    plt.close()
    print("Saved: psc_by_experiment.svg")


def plot_min_distance_by_experiment(df: pd.DataFrame, output_dir: str):
    """Bar chart: Min distance by experiment and scenario."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = df.pivot_table(
        values='min_distance', 
        index='scenario_display', 
        columns='experiment_display', 
        aggfunc='mean'
    )
    
    # Reorder columns
    exp_order = get_experiment_order(df)
    pivot = pivot[[c for c in exp_order if c in pivot.columns]]
    
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black',
               color=get_colors_for_experiments(pivot.columns))
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Collision threshold')
    ax.set_ylabel('Minimum Distance (m, edge-to-edge)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('Minimum Distance by Experiment and Scenario (Mean)', fontsize=14)
    ax.legend(title='Experiment', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'min_distance_by_experiment.svg'))
    plt.close()
    print("Saved: min_distance_by_experiment.svg")


def plot_min_distance_boxplot(df: pd.DataFrame, output_dir: str):
    """Box plot: Min distance distribution by experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_order = get_experiment_order(df)
    data = [df[df['experiment_display'] == exp]['min_distance'].values for exp in exp_order]
    
    bp = ax.boxplot(data, tick_labels=exp_order, patch_artist=True)
    
    colors = get_colors_for_experiments(exp_order)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Collision threshold')
    ax.set_ylabel('Minimum Distance (m, edge-to-edge)', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Minimum Distance Distribution by Experiment', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'min_distance_boxplot.svg'))
    plt.close()
    print("Saved: min_distance_boxplot.svg")


def plot_collision_rate(df: pd.DataFrame, output_dir: str):
    """Bar chart: Collision rate by experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    collision_rate = df.groupby('experiment_display')['collision'].mean() * 100
    
    # Reorder
    exp_order = get_experiment_order(df)
    collision_rate = collision_rate.reindex(exp_order)
    
    colors = get_colors_for_experiments(exp_order)
    bars = ax.bar(collision_rate.index, collision_rate.values, 
                  color=colors, edgecolor='black')
    
    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Collision Rate by Experiment', fontsize=14)
    ax.set_ylim(0, max(100, collision_rate.max() * 1.2))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, collision_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_rate.svg'))
    plt.close()
    print("Saved: collision_rate.svg")


def plot_psc_boxplot(df: pd.DataFrame, output_dir: str):
    """Box plot: PSC distribution by experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_order = get_experiment_order(df)
    data = [df[df['experiment_display'] == exp]['psc'].values for exp in exp_order]
    
    bp = ax.boxplot(data, tick_labels=exp_order, patch_artist=True)
    
    colors = get_colors_for_experiments(exp_order)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Personal Space Compliance (%)', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('PSC Distribution by Experiment', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psc_boxplot.svg'))
    plt.close()
    print("Saved: psc_boxplot.svg")


def plot_vlm_latency_boxplot(df: pd.DataFrame, output_dir: str):
    """Box plot: VLM latency distribution by experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN values
    df_valid = df[df['vlm_latency'].notna()]
    exp_order = [e for e in get_experiment_order(df) if e in df_valid['experiment_display'].unique()]
    data = [df_valid[df_valid['experiment_display'] == exp]['vlm_latency'].values 
            for exp in exp_order]
    
    if all(len(d) == 0 for d in data):
        print("Skipping VLM latency boxplot: no valid data")
        return
    
    bp = ax.boxplot(data, tick_labels=exp_order, patch_artist=True)
    
    colors = get_colors_for_experiments(exp_order)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('VLM Latency (seconds)', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('VLM Response Latency by Experiment', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vlm_latency_boxplot.svg'))
    plt.close()
    print("Saved: vlm_latency_boxplot.svg")


def plot_vlm_actions_stacked(df: pd.DataFrame, output_dir: str):
    """Stacked bar chart: VLM action counts by experiment."""
    action_cols = ['continue_count', 'slow_down_count', 'yield_count']
    
    # Check if columns exist
    if not all(col in df.columns for col in action_cols):
        print("Skipping VLM actions plot: action columns not found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sum actions per experiment
    action_sums = df.groupby('experiment_display')[action_cols].sum()
    
    # Reorder
    exp_order = get_experiment_order(df)
    action_sums = action_sums.reindex(exp_order)
    
    # Create stacked bar chart
    bottom = np.zeros(len(action_sums))
    action_colors = ['#27ae60', '#f39c12', '#e74c3c']
    labels = ['Continue', 'Slow Down', 'Yield']
    
    for col, color, label in zip(action_cols, action_colors, labels):
        ax.bar(action_sums.index, action_sums[col], bottom=bottom, 
               label=label, color=color, edgecolor='black')
        bottom += action_sums[col].values
    
    ax.set_ylabel('Total VLM Actions', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('VLM Action Distribution by Experiment', fontsize=14)
    ax.legend(title='Action Type', loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vlm_actions_stacked.svg'))
    plt.close()
    print("Saved: vlm_actions_stacked.svg")


def plot_scenario_comparison(df: pd.DataFrame, output_dir: str):
    """Heatmap: Average PSC for each experiment-scenario combination."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot_table(
        values='psc', 
        index='experiment_display', 
        columns='scenario_display', 
        aggfunc='mean'
    )
    
    # Reorder
    exp_order = get_experiment_order(df)
    pivot = pivot.reindex(exp_order)
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('PSC (%)', fontsize=12)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Use black text for mid-range values (yellow background)
                # Use white only for very low (red) or very high (green) values
                color = 'black' if 30 < val < 85 else 'white'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('PSC Heatmap: Experiment × Scenario', fontsize=14)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psc_heatmap.svg'))
    plt.close()
    print("Saved: psc_heatmap.svg")


def compute_statistics(df: pd.DataFrame, output_dir: str):
    """Compute and save statistical summary."""
    stats_lines = []
    stats_lines.append("=" * 60)
    stats_lines.append("STATISTICAL SUMMARY")
    stats_lines.append("=" * 60)
    
    # Summary statistics per experiment
    stats_lines.append("\n1. SUMMARY STATISTICS BY EXPERIMENT")
    stats_lines.append("-" * 40)
    
    exp_order = get_experiment_order(df)
    
    for exp in exp_order:
        exp_df = df[df['experiment_display'] == exp]
        stats_lines.append(f"\n{exp.upper()}:")
        stats_lines.append(f"  Scenarios: {len(exp_df)}")
        stats_lines.append(f"  PSC: {exp_df['psc'].mean():.2f}% ± {exp_df['psc'].std():.2f}%")
        stats_lines.append(f"  Min Distance: {exp_df['min_distance'].mean():.3f}m ± {exp_df['min_distance'].std():.3f}m")
        stats_lines.append(f"  Collisions: {exp_df['collision'].sum()} / {len(exp_df)} ({exp_df['collision'].mean()*100:.1f}%)")
        
        if 'vlm_latency' in exp_df.columns:
            vlm_lat = exp_df['vlm_latency'].dropna()
            if len(vlm_lat) > 0:
                stats_lines.append(f"  VLM Latency: {vlm_lat.mean():.3f}s ± {vlm_lat.std():.3f}s")
        
        # Action counts
        if 'continue_count' in exp_df.columns:
            stats_lines.append(f"  Actions - Continue: {exp_df['continue_count'].sum()}, "
                             f"Slow: {exp_df['slow_down_count'].sum()}, "
                             f"Yield: {exp_df['yield_count'].sum()}")
    
    # Statistical tests
    stats_lines.append("\n" + "=" * 60)
    stats_lines.append("2. STATISTICAL TESTS")
    stats_lines.append("-" * 40)
    
    # Kruskal-Wallis test for PSC across experiments
    if len(exp_order) >= 2:
        groups = [df[df['experiment_display'] == exp]['psc'].values for exp in exp_order]
        try:
            stat, p_value = stats.kruskal(*groups)
            stats_lines.append(f"\nKruskal-Wallis test (PSC across experiments):")
            stats_lines.append(f"  H-statistic: {stat:.4f}")
            stats_lines.append(f"  p-value: {p_value:.4f}")
            stats_lines.append(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        except Exception as e:
            stats_lines.append(f"\nKruskal-Wallis test failed: {e}")
    
    # Pairwise Mann-Whitney U tests
    stats_lines.append("\nMann-Whitney U tests (pairwise PSC comparisons):")
    for i, exp1 in enumerate(exp_order):
        for exp2 in exp_order[i+1:]:
            group1 = df[df['experiment_display'] == exp1]['psc'].values
            group2 = df[df['experiment_display'] == exp2]['psc'].values
            try:
                stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                sig = '*' if p_value < 0.05 else ''
                stats_lines.append(f"  {exp1} vs {exp2}: U={stat:.1f}, p={p_value:.4f} {sig}")
            except Exception as e:
                stats_lines.append(f"  {exp1} vs {exp2}: Error - {e}")
    
    # Chi-square test for collision rates
    stats_lines.append("\nChi-square test (collision rates):")
    try:
        contingency = pd.crosstab(df['experiment_display'], df['collision'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        stats_lines.append(f"  χ²: {chi2:.4f}")
        stats_lines.append(f"  p-value: {p_value:.4f}")
        stats_lines.append(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    except Exception as e:
        stats_lines.append(f"  Error: {e}")
    
    # Write to file
    stats_text = '\n'.join(stats_lines)
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w') as f:
        f.write(stats_text)
    
    print(f"Saved: statistics.txt")
    print("\n" + stats_text)


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots and statistics from metrics CSV'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input CSV file with metrics'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots/)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_and_preprocess(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Experiments: {df['experiment_display'].unique().tolist()}")
    print(f"Scenarios: {df['scenario_display'].unique().tolist()}")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_psc_by_experiment(df, args.output_dir)
    plot_min_distance_by_experiment(df, args.output_dir)
    plot_min_distance_boxplot(df, args.output_dir)
    plot_collision_rate(df, args.output_dir)
    plot_psc_boxplot(df, args.output_dir)
    plot_vlm_latency_boxplot(df, args.output_dir)
    plot_vlm_actions_stacked(df, args.output_dir)
    plot_scenario_comparison(df, args.output_dir)
    
    # Compute statistics
    print("\nComputing statistics...")
    compute_statistics(df, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}/")
    print('='*60)


if __name__ == '__main__':
    main()
