#!/usr/bin/env python3
"""
Run metrics extraction on all rosbags in a directory.

Usage (inside Docker):
    python3 run_extraction.py --input /rosbags --output metrics.csv

This script:
1. Finds all rosbag directories in the input folder (ignores subdirectories like 'old')
2. Extracts metrics from each bag using extract_metrics.py
3. Combines all metrics into a single CSV file
"""

import argparse
import os
import sys
from extract_metrics import MetricsExtractor, find_bags_in_directory
from metrics_types import write_metrics_csv


def main():
    parser = argparse.ArgumentParser(
        description='Extract metrics from all rosbags in a directory'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Directory containing rosbag folders (e.g., /rosbags)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='all_metrics.csv',
        help='Output CSV file path (default: all_metrics.csv)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a directory")
        sys.exit(1)
    
    # Find all rosbag directories (only top-level, ignore subdirs like 'old')
    bags = []
    for entry in sorted(os.listdir(args.input)):
        entry_path = os.path.join(args.input, entry)
        
        # Skip if not a directory
        if not os.path.isdir(entry_path):
            continue
        
        # Skip known subdirectories that contain old/archived bags
        if entry.lower() in ['old', 'archive', 'backup']:
            print(f"Skipping subdirectory: {entry}")
            continue
        
        # Check if it contains a .db3 file (is a rosbag)
        for f in os.listdir(entry_path):
            if f.endswith('.db3'):
                bags.append(entry_path)
                break
    
    if not bags:
        print(f"No rosbags found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(bags)} rosbags to process:")
    for bag in bags:
        print(f"  - {os.path.basename(bag)}")
    print()
    
    # Extract metrics from each bag
    all_metrics = []
    for bag_path in bags:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(bag_path)}")
        print('='*60)
        
        try:
            extractor = MetricsExtractor(bag_path)
            metrics = extractor.extract_metrics()
            all_metrics.extend(metrics)
            print(f"Extracted {len(metrics)} scenario metrics")
        except Exception as e:
            print(f"Error processing {bag_path}: {e}")
            continue
    
    # Write combined CSV
    if all_metrics:
        write_metrics_csv(all_metrics, args.output)
        print(f"\n{'='*60}")
        print(f"COMPLETE: Processed {len(bags)} bags")
        print(f"Total scenario metrics: {len(all_metrics)}")
        print(f"Output saved to: {args.output}")
        print('='*60)
    else:
        print("No metrics extracted!")
        sys.exit(1)


if __name__ == '__main__':
    main()
