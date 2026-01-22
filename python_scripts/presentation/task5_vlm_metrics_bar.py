#!/usr/bin/env python3
"""
Task 5: Bar Chart of Grounding, Consistency, and Appropriateness

Creates a bar chart showing:
- Grounding (human and LLM)
- Consistency (human and LLM)
- Appropriateness (calculated)

Data is parsed from vlm_metrics_summary.txt following the code pattern
from summarize_vlm_metrics.py.
"""

import os
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# Presentation color scheme
PRESENTATION_COLORS = {
    'Cyan': '#01ffff',
    'Blue': '#0a00e9',
    'Green': '#3fff45',
    'Pink': '#fc38db',
    'Black': '#000000',
    'White': '#ffffff'
}

# Experiment colors
EXPERIMENT_COLORS = {
    'MISTRAL': PRESENTATION_COLORS['Blue'],
    'SMOL': PRESENTATION_COLORS['Cyan']
}

# Metric colors
METRIC_COLORS = {
    'Grounding (Human)': PRESENTATION_COLORS['Green'],
    'Grounding (LLM)': PRESENTATION_COLORS['Cyan'],
    'Consistency (Human)': PRESENTATION_COLORS['Green'],
    'Consistency (LLM)': PRESENTATION_COLORS['Cyan'],
    'Appropriateness': PRESENTATION_COLORS['Blue']
}


def parse_vlm_metrics_summary(txt_path: str) -> dict:
    """
    Parse VLM metrics summary file.
    
    Returns dict with structure:
    {
        'MISTRAL': {
            'Human G Rate': 50.0,
            'LLM G Rate': 26.2,
            'Human C Rate': 75.0,
            'LLM C Rate': 99.6,
            'Approp Rate': 34.2
        },
        'SMOL': { ... }
    }
    """
    with open(txt_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    current_experiment = None
    
    for line in content.split('\n'):
        # Stop parsing experiment metrics at separator or TOTAL section
        if line.startswith('---') or 'TOTAL' in line:
            current_experiment = None
            continue
        
        # Match experiment headers (MISTRAL, SMOL) - must be preceded by uppercase only
        exp_match = re.match(r'^(MISTRAL|SMOL)$', line.strip())
        if exp_match:
            current_experiment = exp_match.group(1)
            metrics[current_experiment] = {}
            continue
        
        # Match metric lines only for MISTRAL and SMOL (not TOTAL)
        if current_experiment and current_experiment in ('MISTRAL', 'SMOL'):
            # Human G Rate
            match = re.match(r'\s*Human G Rate:\s*([\d.]+)%', line)
            if match:
                metrics[current_experiment]['Human G Rate'] = float(match.group(1))
            
            # LLM G Rate
            match = re.match(r'\s*LLM G Rate:\s*([\d.]+)%', line)
            if match:
                metrics[current_experiment]['LLM G Rate'] = float(match.group(1))
            
            # Human C Rate
            match = re.match(r'\s*Human C Rate:\s*([\d.]+)%', line)
            if match:
                metrics[current_experiment]['Human C Rate'] = float(match.group(1))
            
            # LLM C Rate
            match = re.match(r'\s*LLM C Rate:\s*([\d.]+)%', line)
            if match:
                metrics[current_experiment]['LLM C Rate'] = float(match.group(1))
            
            # Approp Rate
            match = re.match(r'\s*Approp Rate:\s*([\d.]+)%', line)
            if match:
                metrics[current_experiment]['Approp Rate'] = float(match.group(1))
    
    return metrics


def plot_vlm_metrics(metrics: dict, output_path: str):
    """Create bar chart of VLM metrics."""
    experiments = ['MISTRAL', 'SMOL']
    metric_names = ['Grounding (Human)', 'Grounding (Machine)', 
                    'Consistency (Human)', 'Consistency (Machine)', 
                    'Appropriateness']
    
    # Map metric names to dictionary keys
    key_mapping = {
        'Grounding (Human)': 'Human G Rate',
        'Grounding (Machine)': 'LLM G Rate',
        'Consistency (Human)': 'Human C Rate',
        'Consistency (Machine)': 'LLM C Rate',
        'Appropriateness': 'Approp Rate'
    }
    
    # Prepare data
    x = np.arange(len(metric_names))
    width = 0.35
    
    mistral_values = [metrics['MISTRAL'].get(key_mapping[name], 0) for name in metric_names]
    smol_values = [metrics['SMOL'].get(key_mapping[name], 0) for name in metric_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars without black edges
    bars1 = ax.bar(x - width/2, mistral_values, width, 
                   label='Cloud VLM', color=EXPERIMENT_COLORS['MISTRAL'],
                   edgecolor='none', linewidth=0)
    bars2 = ax.bar(x + width/2, smol_values, width, 
                   label='Local VLM', color=EXPERIMENT_COLORS['SMOL'],
                   edgecolor='none', linewidth=0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('VLM Evaluation Metrics: Grounding, Consistency, and Appropriateness', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, '../rosbag_analysis/final_data/vlm_metrics_summary.txt')
    output_path = os.path.join(script_dir, 'vlm_metrics_bar.svg')
    
    # Parse data
    print(f"Parsing VLM metrics from {txt_path}...")
    metrics = parse_vlm_metrics_summary(txt_path)
    
    print("\nParsed metrics:")
    for exp, exp_metrics in metrics.items():
        print(f"\n{exp}:")
        for key, value in exp_metrics.items():
            print(f"  {key}: {value}%")
    
    # Generate plot
    print("\nGenerating VLM metrics bar chart...")
    plot_vlm_metrics(metrics, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
