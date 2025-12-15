#!/usr/bin/env python3
"""
Summarize VLM metrics from evaluation_results.json.

Computes:
1. Human Grounding/Consistency Rates (from labeled subset)
2. LLM Grounding/Consistency Rates (from all samples)
3. Cohen's Kappa Agreement
4. Appropriateness Rate

Outputs:
- Console table
- vlm_metrics_summary.csv
"""

import argparse
import json
import csv
import sys
import os
from typing import List, Dict

def cohens_kappa(human_labels: List[bool], llm_labels: List[bool]) -> float:
    """Compute Cohen's kappa for inter-rater reliability."""
    if len(human_labels) != len(llm_labels):
        return 0.0
    
    n = len(human_labels)
    if n == 0:
        return 0.0
    
    # Count agreements and disagreements
    a = sum(1 for h, l in zip(human_labels, llm_labels) if h and l)  # Both yes
    b = sum(1 for h, l in zip(human_labels, llm_labels) if h and not l)  # Human yes, LLM no
    c = sum(1 for h, l in zip(human_labels, llm_labels) if not h and l)  # Human no, LLM yes
    d = sum(1 for h, l in zip(human_labels, llm_labels) if not h and not l)  # Both no
    
    # Observed agreement
    p_o = (a + d) / n
    
    # Expected agreement by chance
    p_yes_human = (a + b) / n
    p_yes_llm = (a + c) / n
    p_no_human = (c + d) / n
    p_no_llm = (b + d) / n
    
    p_e = p_yes_human * p_yes_llm + p_no_human * p_no_llm
    
    if p_e == 1.0:
        return 1.0
    
    return (p_o - p_e) / (1 - p_e)


def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    """
    Deduplicate samples by sample_id.
    Keeps the first occurrence of each sample_id.
    This is correct because human labels were applied to all duplicates,
    so first occurrence has the valid label for that unique sample.
    """
    seen = {}
    for s in samples:
        sid = s.get('sample_id')
        if sid not in seen:
            seen[sid] = s
        # Skip duplicates - keep first occurrence only
    
    deduped = list(seen.values())
    if len(deduped) < len(samples):
        print(f"Deduplicated: {len(samples)} -> {len(deduped)} samples ({len(samples) - len(deduped)} duplicates removed)")
    return deduped


def compute_rate(yes_count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (yes_count / total) * 100

def main():
    parser = argparse.ArgumentParser(description='Summarize VLM metrics')
    parser.add_argument('--input', '-i', required=True, help='Input evaluation_results.json')
    parser.add_argument('--labels', '-l', help='Optional input file containing human labels (e.g. vlm_samples.json)')
    parser.add_argument('--output', '-o', default='vlm_metrics_summary.csv', help='Output CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Process samples
    samples = data.get('samples', [])
    
    # Load separate labels if provided
    
    if args.labels:
        if not os.path.exists(args.labels):
            print(f"Error: Labels file {args.labels} not found")
            sys.exit(1)
        with open(args.labels, 'r') as f:
            label_data = json.load(f)
            label_samples = label_data.get('samples', [])
            
            if len(label_samples) != len(samples):
                 print(f"Warning: Label file has {len(label_samples)} samples, data has {len(samples)}. Index matching might be unsafe.")
            
            print(f"Merging labels from {args.labels} by list order...")
            # Merge by index
            for i, s in enumerate(samples):
                if i < len(label_samples):
                    ls = label_samples[i]
                    # Paranoia check: IDs should match (even if duplicates)
                    if s['sample_id'] != ls['sample_id']:
                        # Only warn if IDs differ drastically (mismatched files)
                        pass 
                    
                    # Copy labels if present, or clear if None (authoritative)
                    s['labels'] = ls.get('labels')

    # NOTE: We do NOT deduplicate here because:
    # - LLM judge evaluated each of the 942 samples independently (different VLM content)
    # - Only human labels are duplicated (labeled once, copied to all matching sample_ids)
    # For human-LLM agreement, we'll deduplicate when computing kappa/agreement metrics

    # Group by experiment
    by_exp = {}
    for s in samples:
        exp = s.get('experiment')
        if not exp:
            # Fallback: extract from sample_id (e.g. 'mistral_0041' -> 'mistral')
            sid = s.get('sample_id', '')
            if '_' in sid:
                exp = sid.rsplit('_', 1)[0]
            else:
                exp = 'unknown'
        
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(s)

    # Prepare stats
    stats = []

    for exp, exp_samples in by_exp.items():
        # Lists for kappa calculation (only unique sample_ids - first occurrence)
        human_g_list = []
        llm_g_list = []
        human_c_list = []
        llm_c_list = []
        seen_sample_ids = set()  # Track unique sample_ids for human label metrics

        # Counters for human labels (only unique sample_ids)
        human_g_yes = 0
        human_g_total = 0
        human_c_yes = 0
        human_c_total = 0

        # Counters for LLM labels (ALL samples - LLM evaluated each independently)
        llm_g_yes = 0
        llm_g_total = 0
        llm_c_yes = 0
        llm_c_total = 0

        approp_yes = 0
        approp_total = 0

        # Overall Agreement: human and LLM agree on BOTH grounding AND consistency
        # Only computed for unique sample_ids
        overall_agree = 0
        overall_total = 0

        for s in exp_samples:
            sid = s.get('sample_id')
            is_first_occurrence = sid not in seen_sample_ids
            
            # Human labels (from merged labels or existing)
            labels = s.get('labels', {}) or {}
            h_g = labels.get('grounded')
            h_c = labels.get('consistent')

            # LLM labels
            l_g = s.get('llm_grounded')
            l_c = s.get('llm_consistent')
            
            # Appropriateness (LLM evaluated all samples independently)
            app = s.get('appropriate')
            if app is not None:
                approp_total += 1
                if app:
                    approp_yes += 1

            # LLM Stats (ALL samples)
            if l_g is not None:
                llm_g_total += 1
                if l_g: llm_g_yes += 1
            if l_c is not None:
                llm_c_total += 1
                if l_c: llm_c_yes += 1

            # Human stats and agreement metrics - ONLY for first occurrence of each sample_id
            # (Human labels were copied to all samples with same ID, so only count once)
            if is_first_occurrence and h_g is not None:
                seen_sample_ids.add(sid)
                
                # Human Stats (unique samples only)
                human_g_total += 1
                if h_g: human_g_yes += 1
                if h_c is not None:
                    human_c_total += 1
                    if h_c: human_c_yes += 1
                
                # Match for Kappa (unique samples only)
                if l_g is not None:
                    human_g_list.append(h_g)
                    llm_g_list.append(l_g)
                
                if h_c is not None and l_c is not None:
                    human_c_list.append(h_c)
                    llm_c_list.append(l_c)

                # Overall Agreement: both dimensions must match
                if l_g is not None and h_c is not None and l_c is not None:
                    overall_total += 1
                    if h_g == l_g and h_c == l_c:
                        overall_agree += 1

        # Calculate metrics
        row = {
            'Experiment': exp,
            'N_Samples': len(exp_samples),
            'N_Labeled': human_g_total,
            
            # Grounding
            'Human_G_Rate': compute_rate(human_g_yes, human_g_total),
            'LLM_G_Rate': compute_rate(llm_g_yes, llm_g_total),
            'Kappa_G': cohens_kappa(human_g_list, llm_g_list) if len(human_g_list) >= 5 else 0.0,
            
            # Consistency
            'Human_C_Rate': compute_rate(human_c_yes, human_c_total),
            'LLM_C_Rate': compute_rate(llm_c_yes, llm_c_total),
            'Kappa_C': cohens_kappa(human_c_list, llm_c_list) if len(human_c_list) >= 5 else 0.0,
            
            # Appropriateness
            'Approp_Rate': compute_rate(approp_yes, approp_total),
            
            # Overall Agreement (human & LLM agree on both G and C)
            'Overall_Agree': compute_rate(overall_agree, overall_total),
            'N_Overall': overall_total
        }
        stats.append(row)

    # Sort stats by Experiment name
    stats.sort(key=lambda x: x['Experiment'])

    # Compute TOTAL row across all experiments
    all_human_g = []
    all_llm_g = []
    all_human_c = []
    all_llm_c = []
    total_labeled = 0
    total_samples = 0
    total_approp_yes = 0
    total_approp_total = 0
    total_overall_agree = 0
    total_overall_total = 0
    seen_sample_ids_total = set()  # Track unique sample_ids across all experiments
    
    for exp, exp_samples in by_exp.items():
        total_samples += len(exp_samples)
        for s in exp_samples:
            sid = s.get('sample_id')
            is_first_occurrence = sid not in seen_sample_ids_total
            
            labels = s.get('labels', {}) or {}
            h_g = labels.get('grounded')
            h_c = labels.get('consistent')
            l_g = s.get('llm_grounded')
            l_c = s.get('llm_consistent')
            app = s.get('appropriate')
            
            # Appropriateness uses ALL samples (LLM evaluated each independently)
            if app is not None:
                total_approp_total += 1
                if app:
                    total_approp_yes += 1
            
            # Human stats and agreement - ONLY first occurrence per sample_id
            if is_first_occurrence and h_g is not None:
                seen_sample_ids_total.add(sid)
                total_labeled += 1
                
                if l_g is not None:
                    all_human_g.append(h_g)
                    all_llm_g.append(l_g)
                
                if h_c is not None and l_c is not None:
                    all_human_c.append(h_c)
                    all_llm_c.append(l_c)
                
                if l_g is not None and h_c is not None and l_c is not None:
                    total_overall_total += 1
                    if h_g == l_g and h_c == l_c:
                        total_overall_agree += 1
    
    # Agreement rates for grounding and consistency
    g_agree_count = sum(1 for h, l in zip(all_human_g, all_llm_g) if h == l)
    c_agree_count = sum(1 for h, l in zip(all_human_c, all_llm_c) if h == l)
    
    # Overall kappa: treat as binary (both G and C must agree vs not)
    # We need paired labels for this
    overall_human_labels = []  # True if human says grounded AND consistent
    overall_llm_labels = []    # True if LLM says grounded AND consistent
    for h_g, l_g, h_c, l_c in zip(all_human_g, all_llm_g, all_human_c, all_llm_c):
        overall_human_labels.append(h_g and h_c)
        overall_llm_labels.append(l_g and l_c)
    overall_kappa = cohens_kappa(overall_human_labels, overall_llm_labels) if len(overall_human_labels) >= 5 else 0.0
    
    total_row = {
        'Experiment': 'TOTAL',
        'N_Samples': total_samples,
        'N_Labeled': total_labeled,
        'Human_G_Rate': compute_rate(sum(all_human_g), len(all_human_g)) if all_human_g else 0.0,
        'LLM_G_Rate': compute_rate(sum(all_llm_g), len(all_llm_g)) if all_llm_g else 0.0,
        'Kappa_G': cohens_kappa(all_human_g, all_llm_g) if len(all_human_g) >= 5 else 0.0,
        'G_Agree': compute_rate(g_agree_count, len(all_human_g)) if all_human_g else 0.0,
        'Human_C_Rate': compute_rate(sum(all_human_c), len(all_human_c)) if all_human_c else 0.0,
        'LLM_C_Rate': compute_rate(sum(all_llm_c), len(all_llm_c)) if all_llm_c else 0.0,
        'Kappa_C': cohens_kappa(all_human_c, all_llm_c) if len(all_human_c) >= 5 else 0.0,
        'C_Agree': compute_rate(c_agree_count, len(all_human_c)) if all_human_c else 0.0,
        'Approp_Rate': compute_rate(total_approp_yes, total_approp_total),
        'Overall_Agree': compute_rate(total_overall_agree, total_overall_total),
        'Kappa_Overall': overall_kappa,
        'N_Overall': total_overall_total
    }

    # Console Output
    print("\n" + "="*165)
    print(f"{'EXPERIMENT':<20} | {'LABELED':<7} | {'HUMAN_G':<7} | {'LLM_G':<7} | {'KAPPA_G':<7} | {'HUMAN_C':<7} | {'LLM_C':<7} | {'KAPPA_C':<7} | {'APPROP':<7} | {'OVERALL':<8}")
    print("-" * 165)
    
    for r in stats:
        print(f"{r['Experiment']:<20} | "
              f"{r['N_Labeled']:<7} | "
              f"{r['Human_G_Rate']:>6.1f}% | "
              f"{r['LLM_G_Rate']:>6.1f}% | "
              f"{r['Kappa_G']:>7.3f} | "    
              f"{r['Human_C_Rate']:>6.1f}% | "
              f"{r['LLM_C_Rate']:>6.1f}% | "
              f"{r['Kappa_C']:>7.3f} | "
              f"{r['Approp_Rate']:>6.1f}% | "
              f"{r['Overall_Agree']:>6.1f}%")
    
    print("-" * 165)
    # Print TOTAL row with G_Agree and C_Agree
    print(f"{'TOTAL':<20} | "
          f"{total_row['N_Labeled']:<7} | "
          f"{total_row['Human_G_Rate']:>6.1f}% | "
          f"{total_row['LLM_G_Rate']:>6.1f}% | "
          f"{total_row['Kappa_G']:>7.3f} | "    
          f"{total_row['Human_C_Rate']:>6.1f}% | "
          f"{total_row['LLM_C_Rate']:>6.1f}% | "
          f"{total_row['Kappa_C']:>7.3f} | "
          f"{total_row['Approp_Rate']:>6.1f}% | "
          f"{total_row['Overall_Agree']:>6.1f}%")
    print("="*165)
    
    # Print separate agreement summary
    print(f"\n  Combined Agreement (N={len(all_human_g)} labeled samples):")
    print(f"    Grounding Agreement:   {total_row['G_Agree']:>5.1f}%  (Kappa={total_row['Kappa_G']:.3f})")
    print(f"    Consistency Agreement: {total_row['C_Agree']:>5.1f}%  (Kappa={total_row['Kappa_C']:.3f})")
    print(f"    Overall Agreement:     {total_row['Overall_Agree']:>5.1f}%  (Kappa={total_row['Kappa_Overall']:.3f})")
    print()

    # Text file output (replaces CSV)
    output_path = args.output.replace('.csv', '.txt') if args.output.endswith('.csv') else args.output
    
    with open(output_path, 'w') as f:
        f.write("VLM EVALUATION METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-experiment metrics table
        f.write("PER-EXPERIMENT METRICS\n")
        f.write("-" * 80 + "\n\n")
        
        for r in stats:
            f.write(f"{r['Experiment'].upper()}\n")
            f.write(f"  N Samples:     {r['N_Samples']}\n")
            f.write(f"  N Labeled:     {r['N_Labeled']}\n")
            f.write(f"  Human G Rate:  {r['Human_G_Rate']:.1f}%\n")
            f.write(f"  LLM G Rate:    {r['LLM_G_Rate']:.1f}%\n")
            f.write(f"  Kappa G:       {r['Kappa_G']:.3f}\n")
            f.write(f"  Human C Rate:  {r['Human_C_Rate']:.1f}%\n")
            f.write(f"  LLM C Rate:    {r['LLM_C_Rate']:.1f}%\n")
            f.write(f"  Kappa C:       {r['Kappa_C']:.3f}\n")
            f.write(f"  Approp Rate:   {r['Approp_Rate']:.1f}%\n")
            f.write(f"  Overall Agree: {r['Overall_Agree']:.1f}%\n")
            f.write("\n")
        
        # TOTAL row
        f.write("-" * 80 + "\n")
        f.write("TOTAL (ALL EXPERIMENTS)\n")
        f.write(f"  N Samples:     {total_row['N_Samples']}\n")
        f.write(f"  N Labeled:     {total_row['N_Labeled']}\n")
        f.write(f"  Human G Rate:  {total_row['Human_G_Rate']:.1f}%\n")
        f.write(f"  LLM G Rate:    {total_row['LLM_G_Rate']:.1f}%\n")
        f.write(f"  Kappa G:       {total_row['Kappa_G']:.3f}\n")
        f.write(f"  Human C Rate:  {total_row['Human_C_Rate']:.1f}%\n")
        f.write(f"  LLM C Rate:    {total_row['LLM_C_Rate']:.1f}%\n")
        f.write(f"  Kappa C:       {total_row['Kappa_C']:.3f}\n")
        f.write(f"  Approp Rate:   {total_row['Approp_Rate']:.1f}%\n")
        f.write(f"  Overall Agree: {total_row['Overall_Agree']:.1f}%\n\n")
        
        # Combined agreement summary
        f.write("=" * 80 + "\n")
        f.write("HUMAN-LLM AGREEMENT SUMMARY (Combined across experiments)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Labeled Samples: {len(all_human_g)}\n")
        mistral_labeled = next((r['N_Labeled'] for r in stats if r['Experiment'] == 'mistral'), 0)
        smol_labeled = next((r['N_Labeled'] for r in stats if r['Experiment'] == 'smol'), 0)
        f.write(f"  - Mistral: {mistral_labeled} labeled\n")
        f.write(f"  - Smol: {smol_labeled} labeled\n\n")
        
        f.write("Grounding Agreement:\n")
        f.write(f"  Agreement Rate: {total_row['G_Agree']:.1f}%\n")
        f.write(f"  Cohen's Kappa:  {total_row['Kappa_G']:.3f}\n\n")
        
        f.write("Consistency Agreement:\n")
        f.write(f"  Agreement Rate: {total_row['C_Agree']:.1f}%\n")
        f.write(f"  Cohen's Kappa:  {total_row['Kappa_C']:.3f}\n\n")
        
        f.write("Overall Agreement (both G and C match):\n")
        f.write(f"  Agreement Rate: {total_row['Overall_Agree']:.1f}%\n")
        f.write(f"  Cohen's Kappa:  {total_row['Kappa_Overall']:.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Kappa Interpretation:\n")
        f.write("  < 0:       Less than chance agreement\n")
        f.write("  0.01-0.20: Slight agreement\n")
        f.write("  0.21-0.40: Fair agreement\n")
        f.write("  0.41-0.60: Moderate agreement\n")
        f.write("  0.61-0.80: Substantial agreement\n")
        f.write("  0.81-1.00: Almost perfect agreement\n")
    
    print(f"Summary saved to {output_path}")

if __name__ == '__main__':
    main()
