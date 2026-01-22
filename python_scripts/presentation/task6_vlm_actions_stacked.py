#!/usr/bin/env python3
"""
Task 6: VLM Actions Stacked Bar with Presentation Colors

Creates a stacked bar chart showing VLM action distribution by experiment.
Uses presentation color scheme.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# Presentation color scheme
PRESENTATION_COLORS = {
    'Cyan': '#01ffff',
    'Blue': '#0a00e9',
    'Green': '#3fff45',
    'Pink': '#fc38db',
    'Black': '#000000',
    'White': '#ffffff'
}

# Experiment display names and colors
EXPERIMENT_CONFIG = {
    'mistral': {'name': 'Cloud VLM', 'color': PRESENTATION_COLORS['Blue']},
    'smol': {'name': 'Local VLM', 'color': PRESENTATION_COLORS['Cyan']},
    'novlm': {'name': 'No VLM', 'color': PRESENTATION_COLORS['Green']},
    'notrack': {'name': 'No Tracking', 'color': PRESENTATION_COLORS['Pink']}
}

# Action colors (presentation scheme)
ACTION_COLORS = {
    'Continue': PRESENTATION_COLORS['Green'],
    'Slow Down': PRESENTATION_COLORS['Cyan'],
    'Yield': PRESENTATION_COLORS['Pink']
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the metrics CSV."""
    df = pd.read_csv(csv_path)
    
    # Map experiment names to display names
    df['experiment_display'] = df['experiment'].map(
        lambda x: EXPERIMENT_CONFIG.get(x, {}).get('name', x)
    )
    
    return df


def plot_vlm_actions_stacked(df: pd.DataFrame, output_path: str):
    """Create stacked bar chart of VLM actions by experiment."""
    action_cols = ['continue_count', 'slow_down_count', 'yield_count']
    
    # Check if columns exist
    if not all(col in df.columns for col in action_cols):
        print("Skipping VLM actions plot: action columns not found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Sum actions per experiment
    action_sums = df.groupby('experiment_display')[action_cols].sum()
    
    # Only include VLM experiments (Local and Cloud)
    vlm_experiments = ['Local VLM', 'Cloud VLM']
    exp_order = [e for e in vlm_experiments if e in action_sums.index]
    action_sums = action_sums.reindex(exp_order)
    
    if action_sums.empty:
        print("No VLM data found for plotting")
        return
    
    # Create stacked horizontal bar chart
    left = np.zeros(len(action_sums))
    labels = ['Continue', 'Slow Down', 'Yield']
    
    for col, label in zip(action_cols, labels):
        ax.barh(action_sums.index, action_sums[col], left=left, 
                label=label, color=ACTION_COLORS[label], edgecolor='none', linewidth=0)
        left += action_sums[col].values
    
    ax.set_xlabel('Total VLM Actions', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    ax.set_title('VLM Action Distribution by Experiment', fontsize=14, fontweight='bold')
    ax.legend(title='Action Type', loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, exp in enumerate(action_sums.index):
        cumulative = 0
        for col, label in zip(action_cols, labels):
            count = action_sums.loc[exp, col]
            if count > 0:
                # Position label in middle of bar segment
                x_pos = cumulative + count / 2
                ax.text(x_pos, i, str(int(count)), 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            cumulative += count
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print statistics
    print(f"\nVLM Action Statistics:")
    for exp in exp_order:
        if exp in action_sums.index:
            total = action_sums.loc[exp].sum()
            print(f"\n{exp}:")
            print(f"  Total actions: {int(total)}")
            for col, label in zip(action_cols, labels):
                count = action_sums.loc[exp, col]
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {label}: {int(count)} ({pct:.1f}%)")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../rosbag_analysis/final_data/final_metrics.csv')
    output_path = os.path.join(script_dir, 'vlm_actions_stacked.svg')
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Generate plot
    print("\nGenerating VLM actions stacked bar chart...")
    plot_vlm_actions_stacked(df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
