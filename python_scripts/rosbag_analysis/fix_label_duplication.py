#!/usr/bin/env python3
"""
Fix duplicate label application in vlm_samples.json.
Due to ID collisions (multiple bags creating same sample_id), labels were applied to all duplicates.
This script keeps the label on the FIRST occurrence of each ID and removes it from the rest.
"""

import json
import argparse
import shutil
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to vlm_samples.json')
    args = parser.parse_args()
    
    # Backup
    backup_file = args.input_file + '.bak'
    shutil.copy2(args.input_file, backup_file)
    print(f"Backed up to {backup_file}")
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
        
    samples = data.get('samples', [])
    print(f"Loaded {len(samples)} samples")
    
    seen_labeled_ids = set()
    cleaned_count = 0
    kept_count = 0
    
    for s in samples:
        labels = s.get('labels')
        if labels and (labels.get('grounded') is not None or labels.get('consistent') is not None):
            sid = s['sample_id']
            
            if sid in seen_labeled_ids:
                # Duplicate! Remove labels
                s['labels'] = None
                cleaned_count += 1
            else:
                # First time seeing this labeled ID
                seen_labeled_ids.add(sid)
                kept_count += 1
                
    print(f"Cleaned {cleaned_count} duplicate labels.")
    print(f"Kept {kept_count} unique labels.")
    
    with open(args.input_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved fixed file to {args.input_file}")

if __name__ == '__main__':
    main()
