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
        # Lists for kappa calculation
        human_g_list = []
        llm_g_list = []
        human_c_list = []
        llm_c_list = []

        # Counters
        human_g_yes = 0
        human_g_total = 0
        human_c_yes = 0
        human_c_total = 0

        llm_g_yes = 0
        llm_g_total = 0
        llm_c_yes = 0
        llm_c_total = 0

        approp_yes = 0
        approp_total = 0

        for s in exp_samples:
            # Human labels (from merged labels or existing)
            labels = s.get('labels', {}) or {}
            h_g = labels.get('grounded')
            h_c = labels.get('consistent')

            # LLM labels
            l_g = s.get('llm_grounded')
            l_c = s.get('llm_consistent')
            
            # Appropriateness
            app = s.get('appropriate')
            if app is not None:
                approp_total += 1
                if app:
                    approp_yes += 1

            # Match for Kappa?
            if h_g is not None and l_g is not None:
                human_g_list.append(h_g)
                llm_g_list.append(l_g)
            
            if h_c is not None and l_c is not None:
                human_c_list.append(h_c)
                llm_c_list.append(l_c)

            # Human Stats
            if h_g is not None:
                human_g_total += 1
                if h_g: human_g_yes += 1
            if h_c is not None:
                human_c_total += 1
                if h_c: human_c_yes += 1

            # LLM Stats
            if l_g is not None:
                llm_g_total += 1
                if l_g: llm_g_yes += 1
            if l_c is not None:
                llm_c_total += 1
                if l_c: llm_c_yes += 1

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
            'Approp_Rate': compute_rate(approp_yes, approp_total)
        }
        stats.append(row)

    # Sort stats by Experiment name
    stats.sort(key=lambda x: x['Experiment'])

    # Console Output
    print("\n" + "="*145)
    print(f"{'EXPERIMENT':<20} | {'LABELED':<7} | {'HUMAN_G':<7} | {'LLM_G':<7} | {'KAPPA_G':<7} | {'HUMAN_C':<7} | {'LLM_C':<7} | {'KAPPA_C':<7} | {'APPROP':<7}")
    print("-" * 145)
    
    for r in stats:
        print(f"{r['Experiment']:<20} | "
              f"{r['N_Labeled']:<7} | "
              f"{r['Human_G_Rate']:>6.1f}% | "
              f"{r['LLM_G_Rate']:>6.1f}% | "
              f"{r['Kappa_G']:>7.3f} | "    
              f"{r['Human_C_Rate']:>6.1f}% | "
              f"{r['LLM_C_Rate']:>6.1f}% | "
              f"{r['Kappa_C']:>7.3f} | "
              f"{r['Approp_Rate']:>6.1f}%")
        
    print("="*145 + "\n")

    # CSV Output
    fieldnames = ['Experiment', 'N_Samples', 'N_Labeled', 
                  'Human_G_Rate', 'LLM_G_Rate', 'Kappa_G',
                  'Human_C_Rate', 'LLM_C_Rate', 'Kappa_C',
                  'Approp_Rate']
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            # Format floats for CSV
            formatted = row.copy()
            for k, v in formatted.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.3f}"
            writer.writerow(formatted)
            
    print(f"Summary saved to {args.output}")

if __name__ == '__main__':
    main()
