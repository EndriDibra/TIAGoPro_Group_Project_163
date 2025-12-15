#!/usr/bin/env python3
"""
Update only the appropriateness field in evaluation_results.json
without overwriting the LLM judge responses.
"""

import json
from typing import Optional, Tuple


def get_expected_action(distance: Optional[float], ttc: Optional[float]) -> str:
    """
    Determine expected action based on distance and TTC.
    Uses most conservative action when both are available.
    """
    # Default if no data: assume safe to continue
    if distance is None and ttc is None:
        return "Continue"
    
    # Distance-based action
    if distance is not None:
        if distance < 0.45:
            dist_action = "Yield"
        elif distance < 1.2:
            dist_action = "Slow Down"
        else:
            dist_action = "Continue"
    else:
        dist_action = None
    
    # TTC-based action  
    if ttc is not None:
        if ttc < 2.0:
            ttc_action = "Yield"
        elif ttc < 4.0:
            ttc_action = "Slow Down"
        else:
            ttc_action = "Continue"
    else:
        ttc_action = None
    
    # If only one metric available
    if dist_action is None:
        return ttc_action
    if ttc_action is None:
        return dist_action
    
    # Most conservative wins
    priority = {"Yield": 0, "Slow Down": 1, "Continue": 2}
    if priority.get(dist_action, 2) < priority.get(ttc_action, 2):
        return dist_action
    return ttc_action


def check_appropriateness(vlm_action: str, distance: Optional[float], ttc: Optional[float]) -> Tuple[bool, str]:
    """Check if VLM action is appropriate for the context."""
    expected = get_expected_action(distance, ttc)
    
    # Normalize action names
    vlm_action_norm = vlm_action.lower().strip()
    expected_norm = expected.lower().strip()
    
    # Check exact match only (no over-conservativeness allowed)
    is_appropriate = False
    if expected_norm == "yield" and vlm_action_norm in ["yield", "stop"]:
        is_appropriate = True
    elif expected_norm == "slow down" and vlm_action_norm in ["slow down", "slow"]:
        is_appropriate = True
    elif expected_norm == "continue" and vlm_action_norm == "continue":
        is_appropriate = True
    
    reason = f"Expected: {expected}, Got: {vlm_action}"
    return is_appropriate, reason


def main():
    # Load samples
    with open('final_data/vlm_samples.json', 'r') as f:
        samples_data = json.load(f)
    
    # Load existing evaluation results
    with open('final_data/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    # Create lookup from samples
    samples_lookup = {s['sample_id']: s for s in samples_data['samples']}
    
    # Update appropriateness for each sample
    updated_count = 0
    changed_count = 0
    
    for result in eval_results['samples']:
        sample_id = result['sample_id']
        if sample_id not in samples_lookup:
            print(f"Warning: {sample_id} not found in samples")
            continue
        
        sample = samples_lookup[sample_id]
        ctx = sample['context_at_query']
        vlm_action = sample['vlm_response'].get('action', 'Unknown')
        
        distance = ctx.get('min_distance')
        ttc = ctx.get('ttc')
        
        old_appropriate = result.get('appropriate')
        is_appropriate, reason = check_appropriateness(vlm_action, distance, ttc)
        
        result['appropriate'] = is_appropriate
        result['expected_action'] = reason
        updated_count += 1
        
        if old_appropriate != is_appropriate:
            changed_count += 1
    
    # Update metrics aggregation
    for exp_name, exp_metrics in eval_results['metrics_by_experiment'].items():
        # Re-count appropriateness
        exp_samples = [r for r in eval_results['samples'] if r['sample_id'].startswith(exp_name.lower())]
        correct = sum(1 for s in exp_samples if s.get('appropriate'))
        total = sum(1 for s in exp_samples if s.get('appropriate') is not None)
        exp_metrics['appropriateness'] = {'correct': correct, 'total': total}
    
    # Save updated results
    with open('final_data/evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Updated {updated_count} samples")
    print(f"Changed {changed_count} appropriateness values")
    print("Saved to final_data/evaluation_results.json")


if __name__ == '__main__':
    main()
