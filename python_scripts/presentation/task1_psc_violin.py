#!/usr/bin/env python3
"""
Task 1: Violin Plot on PSC for Each Experiment

Creates a violin plot showing PSC distribution for each experiment.
Uses presentation color scheme.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


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

# Consistent order for experiments
EXPERIMENT_ORDER = ['No VLM', 'Local VLM', 'Cloud VLM', 'No Tracking']


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the metrics CSV."""
    df = pd.read_csv(csv_path)
    
    # Map experiment names to display names
    df['experiment_display'] = df['experiment'].map(
        lambda x: EXPERIMENT_CONFIG.get(x, {}).get('name', x)
    )
    
    return df


def plot_psc_violin(df: pd.DataFrame, output_path: str):
    """Create violin plot of PSC distribution for each experiment."""
    # Set seaborn style
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data to only include experiments in EXPERIMENT_ORDER
    df_filtered = df[df['experiment_display'].isin(EXPERIMENT_ORDER)].copy()
    
    # Create color palette in order
    color_palette = []
    for exp_name in EXPERIMENT_ORDER:
        for exp_key, config in EXPERIMENT_CONFIG.items():
            if config['name'] == exp_name:
                color_palette.append(config['color'])
                break
    
    # Create violin plot using seaborn
    sns.violinplot(data=df_filtered, x='experiment_display', y='psc',
                   order=EXPERIMENT_ORDER, palette=color_palette,
                   inner='box', saturation=0.7, ax=ax,
                   cut=0, bw_adjust=0.5)
    
    # Set alpha for violin bodies
    for pc in ax.collections:
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Personal Space Compliance (%)', fontsize=12)
    ax.set_title('PSC Distribution by Experiment', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../rosbag_analysis/final_data/final_metrics.csv')
    output_path = os.path.join(script_dir, 'psc_violin.svg')
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Experiments: {df['experiment_display'].unique().tolist()}")
    
    # Generate plot
    print("\nGenerating PSC violin plot...")
    plot_psc_violin(df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
