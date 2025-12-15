#!/usr/bin/env python3
"""
Interactive labeling tool for VLM explanation samples.

Usage:
    python3 label_samples.py --input vlm_samples.json --output labeled_samples.json

This tool displays VLM samples one-by-one for manual labeling of:
- Grounding: Do detections support the human-related claims?
- Consistency: Does the action logically follow from obs+pred?
"""

import argparse
import json
import os
import sys
from typing import Optional


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_detections(detections: list) -> str:
    """Format detections for display."""
    if not detections:
        return "  (No humans detected)"
    
    lines = []
    for i, d in enumerate(detections):
        angle = d.get('angle', 0)
        # Describe angle direction
        if -15 <= angle <= 15:
            direction = "ahead"
        elif angle > 15:
            direction = "left"
        else:
            direction = "right"
        lines.append(f"  [{i+1}] dist={d['distance']:.2f}m, angle={angle:.0f}° ({direction})")
    return '\n'.join(lines)


def format_sample(sample: dict, idx: int, total: int) -> str:
    """Format a sample for display."""
    vlm = sample['vlm_response']
    ctx = sample['context_at_query']
    
    latency = sample['response_time'] - sample['query_time']
    
    ttc_str = f"{ctx['ttc']:.2f}s" if ctx.get('ttc') else "N/A (not approaching)"
    dist_str = f"{ctx['min_distance']:.2f}m" if ctx.get('min_distance') else "N/A"
    
    # Path interference info
    path_dist = ctx.get('path_distance')
    path_zone = ctx.get('path_zone', 'unknown')
    if path_dist is not None:
        path_str = f"{path_dist:.2f}m ({path_zone})"
    else:
        path_str = "N/A"
    
    return f"""
{'='*70}
Sample {idx+1}/{total} [{sample['sample_id']}]
{'='*70}

VLM RESPONSE:
  Observation: {vlm['observation']}
  
  Prediction:  {vlm['prediction']}
  
  Action:      {vlm['action']}

CONTEXT AT QUERY TIME ({latency:.1f}s before response):
  Detections:
{format_detections(ctx.get('detections', []))}
  
  TTC: {ttc_str}
  Min Distance: {dist_str}
  Path Interference: {path_str}
    (human distance to robot→goal straight line)

{'='*70}
"""


def get_label(prompt: str) -> Optional[bool]:
    """Get a yes/no label from user."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response in ['s', 'skip']:
            return None
        elif response in ['q', 'quit']:
            return 'quit'
        else:
            print("  Please enter: y(es), n(o), s(kip), or q(uit)")


def label_samples(samples: list) -> list:
    """Interactive labeling loop."""
    
    # Filter to only unlabeled samples
    unlabeled = [s for s in samples 
                 if s['labels']['grounded'] is None or s['labels']['consistent'] is None]
    
    if not unlabeled:
        print("All samples are already labeled!")
        return samples
    
    print(f"\nFound {len(unlabeled)} unlabeled samples")
    print("Commands: y=yes, n=no, s=skip, q=quit")
    print("-" * 40)
    input("Press Enter to start labeling...")
    
    labeled_count = 0
    
    for idx, sample in enumerate(unlabeled):
        clear_screen()
        print(format_sample(sample, idx, len(unlabeled)))
        
        # Grounding label
        if sample['labels']['grounded'] is None:
            print("GROUNDING: Do the detections support the human-related claims")
            print("           in the observation and prediction?")
            print("           (Ignore claims about walls or other non-human elements)")
            print()
            result = get_label("  Grounded? (y/n/s/q): ")
            
            if result == 'quit':
                print("\nSaving and quitting...")
                break
            
            sample['labels']['grounded'] = result
        
        print()
        
        # Consistency label
        if sample['labels']['consistent'] is None:
            print("CONSISTENCY: Given the observation and prediction,")
            print("             does the chosen action logically follow?")
            print()
            result = get_label("  Consistent? (y/n/s/q): ")
            
            if result == 'quit':
                print("\nSaving and quitting...")
                break
            
            sample['labels']['consistent'] = result
        
        if sample['labels']['grounded'] is not None or sample['labels']['consistent'] is not None:
            labeled_count += 1
        
        print(f"\n  Saved! [{labeled_count} labeled in this session]")
        
        # Update in main list
        for s in samples:
            if s['sample_id'] == sample['sample_id']:
                s['labels'] = sample['labels']
                break
        
        if idx < len(unlabeled) - 1:
            input("\n  Press Enter for next sample...")
    
    return samples


def compute_stats(samples: list) -> dict:
    """Compute labeling statistics."""
    total = len(samples)
    grounded_yes = sum(1 for s in samples if s['labels'].get('grounded') is True)
    grounded_no = sum(1 for s in samples if s['labels'].get('grounded') is False)
    grounded_skip = sum(1 for s in samples if s['labels'].get('grounded') is None)
    
    consistent_yes = sum(1 for s in samples if s['labels'].get('consistent') is True)
    consistent_no = sum(1 for s in samples if s['labels'].get('consistent') is False)
    consistent_skip = sum(1 for s in samples if s['labels'].get('consistent') is None)
    
    return {
        'total': total,
        'grounded': {
            'yes': grounded_yes,
            'no': grounded_no,
            'skip': grounded_skip,
            'rate': grounded_yes / (grounded_yes + grounded_no) * 100 if (grounded_yes + grounded_no) > 0 else 0
        },
        'consistent': {
            'yes': consistent_yes,
            'no': consistent_no,
            'skip': consistent_skip,
            'rate': consistent_yes / (consistent_yes + consistent_no) * 100 if (consistent_yes + consistent_no) > 0 else 0
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Interactive labeling tool for VLM samples'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file with samples'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON file (default: same as input)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show labeling statistics only'
    )
    parser.add_argument(
        '--max-per-experiment', '-n',
        type=int,
        default=25,
        help='Max samples to label per experiment (default: 25)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    # Load samples
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # Limit samples per experiment for labeling
    from collections import defaultdict
    by_exp = defaultdict(list)
    for s in samples:
        by_exp[s['experiment']].append(s)
    
    samples_to_label = []
    for exp, exp_samples in by_exp.items():
        # Take evenly spaced samples up to max
        n = min(len(exp_samples), args.max_per_experiment)
        if n < len(exp_samples):
            step = len(exp_samples) / n
            selected = [exp_samples[int(i * step)] for i in range(n)]
        else:
            selected = exp_samples
        samples_to_label.extend(selected)
    
    print(f"Selected {len(samples_to_label)} samples for labeling:")
    for exp in by_exp:
        count = len([s for s in samples_to_label if s['experiment'] == exp])
        print(f"  {exp}: {count}")
    
    # Stats only mode
    if args.stats:
        stats = compute_stats(samples)
        print(f"\nLabeling Statistics for {args.input}")
        print("=" * 50)
        print(f"Total samples: {stats['total']}")
        print()
        print("Grounding:")
        print(f"  Yes: {stats['grounded']['yes']}")
        print(f"  No:  {stats['grounded']['no']}")
        print(f"  Skip: {stats['grounded']['skip']}")
        print(f"  Rate: {stats['grounded']['rate']:.1f}%")
        print()
        print("Consistency:")
        print(f"  Yes: {stats['consistent']['yes']}")
        print(f"  No:  {stats['consistent']['no']}")
        print(f"  Skip: {stats['consistent']['skip']}")
        print(f"  Rate: {stats['consistent']['rate']:.1f}%")
        return
    
    # Interactive labeling on selected samples
    labeled_samples = label_samples(samples_to_label)
    
    # Update labels in original samples list
    labeled_by_id = {s['sample_id']: s['labels'] for s in labeled_samples}
    for s in samples:
        if s['sample_id'] in labeled_by_id:
            s['labels'] = labeled_by_id[s['sample_id']]
    
    # Save output (all samples, with updated labels)
    data['samples'] = samples
    output_path = args.output or args.input
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Show final stats
    stats = compute_stats(samples)
    print(f"\n{'='*50}")
    print("FINAL STATISTICS")
    print('='*50)
    print(f"Grounded: {stats['grounded']['yes']}/{stats['grounded']['yes'] + stats['grounded']['no']} ({stats['grounded']['rate']:.1f}%)")
    print(f"Consistent: {stats['consistent']['yes']}/{stats['consistent']['yes'] + stats['consistent']['no']} ({stats['consistent']['rate']:.1f}%)")
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
