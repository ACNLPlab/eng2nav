#!/usr/bin/env python3
"""
Find language code with highest metrics from CSV file.
Usage: python find_best_language.py <filename.csv>
"""

import csv
import sys
from typing import List, Dict

def read_metrics_from_csv(filename: str) -> List[Dict]:
    """Read metrics from CSV file."""
    metrics = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Try to detect if there's a header
            first_line = f.readline()
            f.seek(0)
            
            # Check if first line looks like a header
            has_header = not first_line[0].isdigit() and 'language' in first_line.lower()
            
            reader = csv.reader(f)
            
            if has_header:
                next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 3:
                    try:
                        lang_code = row[0].strip()
                        metric1 = float(row[1].strip())
                        metric2 = float(row[2].strip())
                        
                        metrics.append({
                            'language': lang_code,
                            'metric1': metric1,
                            'metric2': metric2,
                            'combined': metric1 + metric2
                        })
                    except ValueError:
                        print(f"Warning: Skipping invalid row: {row}", file=sys.stderr)
                        continue
                        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return metrics

def find_best_languages(metrics: List[Dict]) -> Dict:
    """Find languages with highest metrics."""
    if not metrics:
        return None
    
    best_metric1 = max(metrics, key=lambda x: x['metric1'])
    best_metric2 = max(metrics, key=lambda x: x['metric2'])
    best_combined = max(metrics, key=lambda x: x['combined'])
    
    top10_metric1 = sorted(metrics, key=lambda x: x['metric1'], reverse=True)[:10]
    top10_metric2 = sorted(metrics, key=lambda x: x['metric2'], reverse=True)[:10]
    top10_combined = sorted(metrics, key=lambda x: x['combined'], reverse=True)[:10]
    
    return {
        'best_metric1': best_metric1,
        'best_metric2': best_metric2,
        'best_combined': best_combined,
        'top10_metric1': top10_metric1,
        'top10_metric2': top10_metric2,
        'top10_combined': top10_combined
    }

def print_results(results: Dict):
    """Print results in a formatted way."""
    print("=" * 70)
    print("BEST SINGLE LANGUAGE BY METRIC")
    print("=" * 70)
    print(f"\nHighest Metric 1 (Column 2):")
    print(f"  {results['best_metric1']['language']}: {results['best_metric1']['metric1']:.5f}")
    
    print(f"\nHighest Metric 2 (Column 3):")
    print(f"  {results['best_metric2']['language']}: {results['best_metric2']['metric2']:.5f}")
    
    print(f"\nHighest Combined Score:")
    print(f"  {results['best_combined']['language']}: {results['best_combined']['combined']:.5f}")
    print(f"  (Metric1: {results['best_combined']['metric1']:.5f}, Metric2: {results['best_combined']['metric2']:.5f})")
    
    print("\n" + "=" * 70)
    print("TOP 10 LANGUAGES BY METRIC 1")
    print("=" * 70)
    for i, lang in enumerate(results['top10_metric1'], 1):
        print(f"{i:2d}. {lang['language']:12s} - {lang['metric1']:8.5f}")
    
    print("\n" + "=" * 70)
    print("TOP 10 LANGUAGES BY METRIC 2")
    print("=" * 70)
    for i, lang in enumerate(results['top10_metric2'], 1):
        print(f"{i:2d}. {lang['language']:12s} - {lang['metric2']:8.5f}")
    
    print("\n" + "=" * 70)
    print("TOP 10 LANGUAGES BY COMBINED SCORE")
    print("=" * 70)
    for i, lang in enumerate(results['top10_combined'], 1):
        print(f"{i:2d}. {lang['language']:12s} - Combined: {lang['combined']:8.5f} "
              f"(M1: {lang['metric1']:.5f}, M2: {lang['metric2']:.5f})")

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_best_language.py <filename.csv>")
        print("\nExample: python find_best_language.py metrics.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"Reading metrics from: {filename}\n")
    
    # Read metrics from CSV
    metrics = read_metrics_from_csv(filename)
    
    if not metrics:
        print("No valid metrics found in file!")
        sys.exit(1)
    
    print(f"Loaded {len(metrics)} language entries.\n")
    
    # Find best languages
    results = find_best_languages(metrics)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    main()
