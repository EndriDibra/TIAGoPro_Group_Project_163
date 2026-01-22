#!/usr/bin/env python3
"""
Task 2: Collision Heatmap with Presentation Colors

Creates a collision heatmap showing collision rate by experiment and scenario.
Uses presentation color scheme.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

# Scenario display names
SCENARIO_NAMES = {
    'frontal_approach': 'Frontal Approach',
    'intersection': 'Intersection',
    'doorway': 'Doorway'
}

# Consistent order for experiments
EXPERIMENT_ORDER = ['No VLM', 'Local VLM', 'Cloud VLM', 'No Tracking']
SCENARIO_ORDER = ['Frontal Approach', 'Intersection', 'Doorway']


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the metrics CSV."""
    df = pd.read_csv(csv_path)
    
    # Map experiment and scenario names to display names
    df['experiment_display'] = df['experiment'].map(
        lambda x: EXPERIMENT_CONFIG.get(x, {}).get('name', x)
    )
    df['scenario_display'] = df['scenario'].map(
        lambda x: SCENARIO_NAMES.get(x, x)
    )
    
    # Add collision indicator
    df['collision'] = df['min_distance'] < 0
    
    return df


def plot_collision_heatmap(df: pd.DataFrame, output_path: str):
    """Create collision rate heatmap with presentation colors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot table for collision rate (mean of boolean represents percentage)
    pivot = df.pivot_table(
        values='collision', 
        index='experiment_display', 
        columns='scenario_display', 
        aggfunc='mean'
    ) * 100  # Convert to percentage
    
    # Reorder
    pivot = pivot.reindex(EXPERIMENT_ORDER)
    pivot = pivot[SCENARIO_ORDER]
    
    # Create custom colormap from white to pink (using presentation colors)
    # White (0%) -> Pink (100%)
    colors_list = [
        PRESENTATION_COLORS['White'],  # 0% collision
        PRESENTATION_COLORS['Pink']    # 100% collision
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('presentation_pink', colors_list)
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Collision Rate (%)', fontsize=12)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Use black text for lighter backgrounds, white for darker pink
                color = 'black' if val < 50 else 'white'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('Collision Rate Heatmap: Experiment × Scenario', fontsize=14)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Experiment', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../rosbag_analysis/final_data/final_metrics.csv')
    output_path = os.path.join(script_dir, 'collision_heatmap.svg')
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Experiments: {df['experiment_display'].unique().tolist()}")
    print(f"Scenarios: {df['scenario_display'].unique().tolist()}")
    
    # Generate plot
    print("\nGenerating collision heatmap...")
    plot_collision_heatmap(df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
