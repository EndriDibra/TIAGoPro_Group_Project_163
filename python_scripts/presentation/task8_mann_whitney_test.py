#!/usr/bin/env python3
"""
Task 8: Mann-Whitney U Test for No VLM vs. No Tracking

Computes Mann-Whitney U test comparing No VLM vs. No Tracking for:
- PSC
- Min Distance
- Collision rate

Provides full precision (no rounding) in the output.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats


# Experiment display names
EXPERIMENT_NAMES = {
    'mistral': 'Cloud VLM',
    'smol': 'Local VLM',
    'novlm': 'No VLM',
    'notrack': 'No Tracking'
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the metrics CSV."""
    df = pd.read_csv(csv_path)
    
    # Map experiment names to display names
    df['experiment_display'] = df['experiment'].map(
        lambda x: EXPERIMENT_NAMES.get(x, x)
    )
    
    # Add collision indicator
    df['collision'] = df['min_distance'] < 0
    
    return df


def compute_mann_whitney_tests(df: pd.DataFrame, output_path: str):
    """
    Compute Mann-Whitney U tests comparing No VLM vs. No Tracking.
    Also compute Kruskal-Wallis test for PSC across all experiments.
    
    Tests for:
    - PSC (Personal Space Compliance)
    - Min Distance
    - Collision rate
    - Kruskal-Wallis test for PSC across all experiments
    """
    # Filter data for the two experiments
    novlm_df = df[df['experiment_display'] == 'No VLM']
    notrack_df = df[df['experiment_display'] == 'No Tracking']
    
    if novlm_df.empty or notrack_df.empty:
        print("Error: Missing data for No VLM or No Tracking experiments")
        return
    
    # Prepare output lines
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("STATISTICAL TESTS FOR PSC AND DISTANCE METRICS")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Kruskal-Wallis test for PSC across all experiments
    output_lines.append("0. KRUSKAL-WALLIS TEST: PSC across all experiments")
    output_lines.append("-" * 80)
    
    all_experiments = df['experiment_display'].unique()
    psc_groups = []
    group_stats = []
    
    for exp in sorted(all_experiments):
        exp_df = df[df['experiment_display'] == exp]
        psc_data = exp_df['psc'].values
        psc_data = psc_data[~np.isnan(psc_data)]
        psc_groups.append(psc_data)
        group_stats.append((exp, len(psc_data), np.mean(psc_data), np.std(psc_data)))
    
    # Print group statistics
    for exp, n, mean, std in group_stats:
        output_lines.append(f"{exp}: n={n}, mean={mean}, std={std}")
    
    try:
        statistic, p_value = stats.kruskal(*psc_groups)
        output_lines.append("")
        output_lines.append("Kruskal-Wallis Test Results:")
        output_lines.append(f"  H statistic: {statistic}")
        output_lines.append(f"  p-value: {p_value}")
        output_lines.append(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (eta-squared)
        # For Kruskal-Wallis: η² = (H - k + 1) / (N - k)
        k = len(psc_groups)
        N = sum(len(group) for group in psc_groups)
        eta_squared = (statistic - k + 1) / (N - k)
        output_lines.append(f"  Effect size (η²): {eta_squared}")
        
        # Interpret effect size
        if eta_squared < 0.01:
            effect_interp = "negligible"
        elif eta_squared < 0.06:
            effect_interp = "small"
        elif eta_squared < 0.14:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        output_lines.append(f"  Effect size interpretation: {effect_interp}")
    except Exception as e:
        output_lines.append(f"  Error computing test: {e}")
    
    output_lines.append("")
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("MANN-WHITNEY U TEST: No VLM vs. No Tracking")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Test 1: PSC (Personal Space Compliance)
    output_lines.append("1. PERSONAL SPACE COMPLIANCE (PSC)")
    output_lines.append("-" * 80)
    psc_novlm = novlm_df['psc'].values
    psc_notrack = notrack_df['psc'].values
    
    # Remove NaN values
    psc_novlm = psc_novlm[~np.isnan(psc_novlm)]
    psc_notrack = psc_notrack[~np.isnan(psc_notrack)]
    
    output_lines.append(f"No VLM: n={len(psc_novlm)}, mean={np.mean(psc_novlm)}, std={np.std(psc_novlm)}")
    output_lines.append(f"No Tracking: n={len(psc_notrack)}, mean={np.mean(psc_notrack)}, std={np.std(psc_notrack)}")
    
    try:
        statistic, p_value = stats.mannwhitneyu(psc_novlm, psc_notrack, alternative='two-sided')
        output_lines.append("")
        output_lines.append("Mann-Whitney U Test Results:")
        output_lines.append(f"  U statistic: {statistic}")
        output_lines.append(f"  p-value: {p_value}")
        output_lines.append(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size (r = Z / sqrt(N))
        # Calculate Z-score from U statistic
        n1, n2 = len(psc_novlm), len(psc_notrack)
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (statistic - mean_u) / std_u
        r_effect = abs(z_score) / np.sqrt(n1 + n2)
        output_lines.append(f"  Effect size (r): {r_effect}")
        
        # Interpret effect size
        if r_effect < 0.1:
            effect_interp = "negligible"
        elif r_effect < 0.3:
            effect_interp = "small"
        elif r_effect < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        output_lines.append(f"  Effect size interpretation: {effect_interp}")
    except Exception as e:
        output_lines.append(f"  Error computing test: {e}")
    
    output_lines.append("")
    output_lines.append("")
    
    # Test 2: Min Distance
    output_lines.append("2. MINIMUM DISTANCE")
    output_lines.append("-" * 80)
    min_dist_novlm = novlm_df['min_distance'].values
    min_dist_notrack = notrack_df['min_distance'].values
    
    # Remove NaN and inf values
    min_dist_novlm = min_dist_novlm[~np.isnan(min_dist_novlm) & ~np.isinf(min_dist_novlm)]
    min_dist_notrack = min_dist_notrack[~np.isnan(min_dist_notrack) & ~np.isinf(min_dist_notrack)]
    
    output_lines.append(f"No VLM: n={len(min_dist_novlm)}, mean={np.mean(min_dist_novlm)}, std={np.std(min_dist_novlm)}")
    output_lines.append(f"No Tracking: n={len(min_dist_notrack)}, mean={np.mean(min_dist_notrack)}, std={np.std(min_dist_notrack)}")
    
    try:
        statistic, p_value = stats.mannwhitneyu(min_dist_novlm, min_dist_notrack, alternative='two-sided')
        output_lines.append("")
        output_lines.append("Mann-Whitney U Test Results:")
        output_lines.append(f"  U statistic: {statistic}")
        output_lines.append(f"  p-value: {p_value}")
        output_lines.append(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Effect size
        n1, n2 = len(min_dist_novlm), len(min_dist_notrack)
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (statistic - mean_u) / std_u
        r_effect = abs(z_score) / np.sqrt(n1 + n2)
        output_lines.append(f"  Effect size (r): {r_effect}")
        
        if r_effect < 0.1:
            effect_interp = "negligible"
        elif r_effect < 0.3:
            effect_interp = "small"
        elif r_effect < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        output_lines.append(f"  Effect size interpretation: {effect_interp}")
    except Exception as e:
        output_lines.append(f"  Error computing test: {e}")
    
    output_lines.append("")
    output_lines.append("")
    
    # Test 3: Collision Rate (binary data - use Fisher's exact test or Chi-square)
    output_lines.append("3. COLLISION RATE")
    output_lines.append("-" * 80)
    
    # Create contingency table
    collision_novlm = novlm_df['collision'].sum()
    no_collision_novlm = len(novlm_df) - collision_novlm
    collision_notrack = notrack_df['collision'].sum()
    no_collision_notrack = len(notrack_df) - collision_notrack
    
    output_lines.append("Contingency Table:")
    output_lines.append(f"              Collision  No Collision  Total")
    output_lines.append(f"No VLM         {collision_novlm}          {no_collision_novlm}       {len(novlm_df)}")
    output_lines.append(f"No Tracking     {collision_notrack}          {no_collision_notrack}       {len(notrack_df)}")
    output_lines.append(f"Total           {collision_novlm + collision_notrack}          {no_collision_novlm + no_collision_notrack}       {len(novlm_df) + len(notrack_df)}")
    
    # Calculate collision rates
    rate_novlm = collision_novlm / len(novlm_df) * 100
    rate_notrack = collision_notrack / len(notrack_df) * 100
    output_lines.append("")
    output_lines.append(f"No VLM collision rate: {rate_novlm}%")
    output_lines.append(f"No Tracking collision rate: {rate_notrack}%")
    
    # Fisher's exact test (better for small samples)
    try:
        contingency_table = [[collision_novlm, no_collision_novlm],
                           [collision_notrack, no_collision_notrack]]
        oddsratio, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
        
        output_lines.append("")
        output_lines.append("Fisher's Exact Test Results:")
        output_lines.append(f"  Odds ratio: {oddsratio}")
        output_lines.append(f"  p-value: {p_value}")
        output_lines.append(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Interpret odds ratio
        if oddsratio > 1:
            output_lines.append(f"  Interpretation: No VLM has {oddsratio:.2f}x higher odds of collision")
        else:
            output_lines.append(f"  Interpretation: No Tracking has {1/oddsratio:.2f}x higher odds of collision")
    except Exception as e:
        output_lines.append(f"  Error computing Fisher's exact test: {e}")
    
    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append("NOTES:")
    output_lines.append("- Kruskal-Wallis test is a non-parametric test for comparing multiple")
    output_lines.append("  independent samples. Effect size (η²): <0.01 negligible, 0.01-0.06")
    output_lines.append("  small, 0.06-0.14 medium, >0.14 large.")
    output_lines.append("- Mann-Whitney U test is a non-parametric test for comparing two")
    output_lines.append("  independent samples when the assumption of normality is violated.")
    output_lines.append("- Fisher's exact test is used for binary (collision) data.")
    output_lines.append("- Effect size (r) interpretation: <0.1 negligible, 0.1-0.3 small,")
    output_lines.append("  0.3-0.5 medium, >0.5 large.")
    output_lines.append("- All values are shown with full precision (no rounding).")
    output_lines.append("=" * 80)
    
    # Write to file
    output_text = '\n'.join(output_lines)
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"Saved: {output_path}")
    print("\n" + output_text)


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../rosbag_analysis/final_data/final_metrics.csv')
    output_path = os.path.join(script_dir, 'mann_whitney_results.txt')
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Compute tests
    print("\nComputing Mann-Whitney U tests...")
    compute_mann_whitney_tests(df, output_path)
    
    print(f"\nDone! Output saved to: {output_path}")


if __name__ == '__main__':
    main()
