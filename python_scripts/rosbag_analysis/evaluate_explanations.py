#!/usr/bin/env python3
"""
Evaluate VLM explanations using LLM judge and rule-based metrics.

Usage:
    python3 evaluate_explanations.py --input labeled_samples.json --output results.json

This script computes:
1. Grounding: LLM judge compares VLM claims vs detections
2. Consistency: LLM judge checks if action follows from obs+pred
3. Appropriateness: Rule-based check using TTC+distance thresholds
4. Cohen's kappa: Agreement between human labels and LLM judge
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set directly

# Optional: Mistral API for LLM judge
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


# =============================================================================
# Appropriateness Thresholds
# =============================================================================

def get_expected_action(distance: Optional[float], ttc: Optional[float]) -> str:
    """
    Determine expected action based on distance and TTC.
    Uses most conservative action when both are available.
    """
    # Default if no data: assume safe to continue
    if distance is None and ttc is None:
        return "Continue"
    
    # Distance-based action
    # Note: These are center-to-center, not accounting for footprints
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


def check_appropriateness(sample: dict) -> Tuple[bool, str]:
    """Check if VLM action is appropriate for the context."""
    ctx = sample['context_at_query']
    vlm_action = sample['vlm_response'].get('action', 'Unknown')
    
    distance = ctx.get('min_distance')
    ttc = ctx.get('ttc')
    
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


# =============================================================================
# LLM Judge
# =============================================================================

GROUNDING_PROMPT = """Evaluate if the VLM's claims about humans are supported by sensor data.

SENSOR DATA (from when VLM was queried):
- Human Detections: {detections}
  (Each detection includes: x, y in meters, distance in meters, angle in degrees from robot forward)
- Time-to-Collision: {ttc}
- Path Interference: {path_info}
  (Human distance to robot->goal line. Zones: intimate <0.45m, personal 0.45-1.2m, clear >1.2m)

VLM OUTPUT:
- Observation: "{observation}"
- Prediction: "{prediction}"

Do the detections support the human-related claims in the observation and prediction?
(Ignore claims about walls or other non-human elements)

Answer with EXACTLY "YES" or "NO" on the first line, then briefly explain."""


CONSISTENCY_PROMPT = """Evaluate if the VLM's action logically follows from its observation and prediction.

VLM OUTPUT:
- Observation: "{observation}"
- Prediction: "{prediction}"
- Action: "{action}"

Given what the VLM observed and predicted about the human, does the chosen action logically follow?

Answer with EXACTLY "YES" or "NO" on the first line, then briefly explain."""


class LLMJudge:
    """LLM-based judge for grounding and consistency."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('MISTRAL_API_KEY')
        self.client = None
        
        if MISTRAL_AVAILABLE and self.api_key:
            try:
                self.client = Mistral(api_key=self.api_key)
                print("LLM Judge initialized with Mistral API")
            except Exception as e:
                print(f"Failed to initialize Mistral: {e}")
        else:
            print("LLM Judge not available (no API key or mistralai not installed)")
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM and get response."""
        if not self.client:
            return None
        
        try:
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Low temp for consistency
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None
    
    def _parse_yes_no(self, response: str) -> Optional[bool]:
        """Parse YES/NO from LLM response."""
        if not response:
            return None
        
        first_line = response.strip().split('\n')[0].upper()
        if 'YES' in first_line:
            return True
        elif 'NO' in first_line:
            return False
        return None
    
    def judge_grounding(self, sample: dict) -> Tuple[Optional[bool], str]:
        """Judge if VLM observation is grounded in detections."""
        ctx = sample['context_at_query']
        vlm = sample['vlm_response']
        
        detections_str = json.dumps(ctx.get('detections', []))
        ttc_str = f"{ctx['ttc']:.2f}s" if ctx.get('ttc') else "N/A"
        
        # Path interference info
        path_dist = ctx.get('path_distance')
        path_zone = ctx.get('path_zone', 'unknown')
        if path_dist is not None:
            path_info = f"{path_dist:.2f}m ({path_zone})"
        else:
            path_info = "N/A"
        
        prompt = GROUNDING_PROMPT.format(
            detections=detections_str,
            ttc=ttc_str,
            path_info=path_info,
            observation=vlm.get('observation', ''),
            prediction=vlm.get('prediction', '')
        )
        
        response = self._call_llm(prompt)
        result = self._parse_yes_no(response)
        return result, response or "LLM unavailable"
    
    def judge_consistency(self, sample: dict) -> Tuple[Optional[bool], str]:
        """Judge if VLM action is consistent with obs+pred."""
        vlm = sample['vlm_response']
        
        prompt = CONSISTENCY_PROMPT.format(
            observation=vlm.get('observation', ''),
            prediction=vlm.get('prediction', ''),
            action=vlm.get('action', '')
        )
        
        response = self._call_llm(prompt)
        result = self._parse_yes_no(response)
        return result, response or "LLM unavailable"


# =============================================================================
# Cohen's Kappa
# =============================================================================

def cohens_kappa(human_labels: List[bool], llm_labels: List[bool]) -> float:
    """
    Compute Cohen's kappa for inter-rater reliability.
    
    κ = (p_o - p_e) / (1 - p_e)
    where:
    - p_o = observed agreement
    - p_e = expected agreement by chance
    """
    if len(human_labels) != len(llm_labels):
        raise ValueError("Label lists must have same length")
    
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
        return 1.0  # Perfect agreement
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.2:
        return "Slight"
    elif kappa < 0.4:
        return "Fair"
    elif kappa < 0.6:
        return "Moderate"
    elif kappa < 0.8:
        return "Substantial"
    else:
        return "Almost Perfect"


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_samples(samples: List[dict], llm_judge: LLMJudge, 
                     use_llm: bool = True) -> dict:
    """Evaluate all samples and compute metrics."""
    
    results = {
        'samples': [],
        'metrics_by_experiment': {},
    }
    
    # Group samples by experiment
    by_exp = {}
    for s in samples:
        exp = s['experiment']
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(s)
    
    for exp, exp_samples in by_exp.items():
        print(f"\nEvaluating {exp}: {len(exp_samples)} samples")
        
        exp_results = {
            'grounding': {'correct': 0, 'total': 0},
            'consistency': {'correct': 0, 'total': 0},
            'appropriateness': {'correct': 0, 'total': 0},
            'kappa': {'grounding': None, 'consistency': None}
        }
        
        human_grounded = []
        llm_grounded = []
        human_consistent = []
        llm_consistent = []
        
        start_time = time.time()
        
        for i, sample in enumerate(exp_samples):
            # Progress bar with ETA
            if i % 5 == 0 or i == len(exp_samples) - 1:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_per_sample = elapsed / i
                    remaining = avg_per_sample * (len(exp_samples) - i)
                    eta_str = f"ETA: {int(remaining//60)}m {int(remaining%60)}s"
                else:
                    eta_str = "Calculating ETA..."
                
                pct = (i + 1) / len(exp_samples) * 100
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = '=' * filled + '-' * (bar_len - filled)
                print(f"\r  [{bar}] {i+1}/{len(exp_samples)} ({pct:.0f}%) | {eta_str}", end='', flush=True)

            sample_result = {
                'sample_id': sample['sample_id'],
                'llm_grounded': None,
                'llm_consistent': None,
                'appropriate': None,
                'expected_action': None
            }
            
            # Appropriateness (rule-based)
            is_appropriate, reason = check_appropriateness(sample)
            if is_appropriate is not None:
                sample_result['appropriate'] = is_appropriate
                sample_result['expected_action'] = reason
                exp_results['appropriateness']['total'] += 1
                if is_appropriate:
                    exp_results['appropriateness']['correct'] += 1
            
            # LLM Judge (if available)
            if use_llm and llm_judge.client:
                # Rate limit (moved logic inside progress bar block above to avoid redundancy)
                # But kept minimal sleep to respect API limits if needed
                # time.sleep(0.1) 
                
                # Grounding
                llm_result, explanation = llm_judge.judge_grounding(sample)
                if llm_result is not None:
                    sample_result['llm_grounded'] = llm_result
                    exp_results['grounding']['total'] += 1
                    if llm_result:
                        exp_results['grounding']['correct'] += 1
                    
                    # For kappa: compare with human label
                    human_label = sample['labels'].get('grounded')
                    if human_label is not None:
                        human_grounded.append(human_label)
                        llm_grounded.append(llm_result)
                
                # Consistency
                llm_result, explanation = llm_judge.judge_consistency(sample)
                if llm_result is not None:
                    sample_result['llm_consistent'] = llm_result
                    exp_results['consistency']['total'] += 1
                    if llm_result:
                        exp_results['consistency']['correct'] += 1
                    
                    human_label = sample['labels'].get('consistent')
                    if human_label is not None:
                        human_consistent.append(human_label)
                        llm_consistent.append(llm_result)
            
            results['samples'].append(sample_result)
        
        # Newline after progress bar
        print()
        
        # Compute kappa if we have matched labels
        if len(human_grounded) >= 5:
            exp_results['kappa']['grounding'] = cohens_kappa(human_grounded, llm_grounded)
        if len(human_consistent) >= 5:
            exp_results['kappa']['consistency'] = cohens_kappa(human_consistent, llm_consistent)
        
        results['metrics_by_experiment'][exp] = exp_results
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "="*70)
    print("VLM EXPLANATION EVALUATION RESULTS")
    print("="*70)
    
    for exp, metrics in results['metrics_by_experiment'].items():
        print(f"\n{exp.upper()}")
        print("-"*40)
        
        # Grounding
        g = metrics['grounding']
        if g['total'] > 0:
            rate = g['correct'] / g['total'] * 100
            print(f"  Grounding:     {g['correct']}/{g['total']} ({rate:.1f}%)")
        
        # Consistency
        c = metrics['consistency']
        if c['total'] > 0:
            rate = c['correct'] / c['total'] * 100
            print(f"  Consistency:   {c['correct']}/{c['total']} ({rate:.1f}%)")
        
        # Appropriateness
        a = metrics['appropriateness']
        if a['total'] > 0:
            rate = a['correct'] / a['total'] * 100
            print(f"  Appropriate:   {a['correct']}/{a['total']} ({rate:.1f}%)")
        
        # Kappa
        k = metrics['kappa']
        if k.get('grounding') is not None:
            kappa = k['grounding']
            print(f"  κ (Grounding): {kappa:.3f} ({interpret_kappa(kappa)})")
        if k.get('consistency') is not None:
            kappa = k['consistency']
            print(f"  κ (Consist.):  {kappa:.3f} ({interpret_kappa(kappa)})")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VLM explanations'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file with labeled samples'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='evaluation_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM judge (only compute appropriateness)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Mistral API key (or set MISTRAL_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    # Load samples
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    
    # Initialize LLM judge
    llm_judge = LLMJudge(api_key=args.api_key)
    
    # Run evaluation
    use_llm = not args.no_llm and llm_judge.client is not None
    results = evaluate_samples(samples, llm_judge, use_llm=use_llm)
    
    # Print results
    print_results(results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
