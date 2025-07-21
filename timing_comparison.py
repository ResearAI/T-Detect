#!/usr/bin/env python3
"""
Timing comparison script for fast_detect_raid vs fast_detect vs binoculars
"""
import time
import json
import sys
import os
sys.path.append('scripts')

from scripts.detectors import get_detector
import numpy as np

def load_sample_data(dataset_path, max_samples=100):
    """Load sample data from dataset for timing tests"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Take first max_samples for consistent testing
    samples = data[:max_samples]
    texts = [item['generation'] for item in samples]
    return texts

def time_detector(detector_name, texts, num_runs=3):
    """Time detector performance on given texts"""
    print(f"\nTesting {detector_name}...")
    
    # Initialize detector
    detector = get_detector(detector_name)
    
    times = []
    for run in range(num_runs):
        start_time = time.time()
        
        # Process all texts
        for text in texts:
            try:
                _ = detector.compute_crit(text)
            except Exception as e:
                print(f"Error processing text with {detector_name}: {e}")
                continue
        
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        print(f"  Run {run + 1}: {run_time:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(texts) / avg_time  # texts per second
    
    return {
        'detector': detector_name,
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': throughput,
        'times': times
    }

def main():
    # Test parameters
    dataset_path = "./benchmark/hart/essay.dev.json"
    max_samples = 100  # Use more samples for better accuracy
    detectors = ['fast_detect', 'fast_detect_raid', 'binoculars']
    
    print(f"Loading sample data from {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    texts = load_sample_data(dataset_path, max_samples)
    print(f"Loaded {len(texts)} texts for timing comparison")
    
    # Run timing tests
    results = []
    for detector_name in detectors:
        try:
            result = time_detector(detector_name, texts)
            results.append(result)
        except Exception as e:
            print(f"Failed to test {detector_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("TIMING COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Detector':<20} {'Avg Time (s)':<12} {'Throughput (texts/s)':<20} {'Std Dev (s)':<12}")
    print("-"*60)
    
    for result in results:
        print(f"{result['detector']:<20} {result['avg_time']:<12.3f} {result['throughput']:<20.2f} {result['std_time']:<12.3f}")
    
    # Calculate relative performance
    if len(results) >= 2:
        print("\n" + "="*60)
        print("RELATIVE PERFORMANCE")
        print("="*60)
        
        # Use fast_detect as baseline
        baseline = next((r for r in results if r['detector'] == 'fast_detect'), None)
        if baseline:
            print(f"Baseline: {baseline['detector']} ({baseline['throughput']:.2f} texts/s)")
            print("-"*60)
            
            for result in results:
                if result['detector'] != 'fast_detect':
                    relative_speed = result['throughput'] / baseline['throughput']
                    speed_change = (relative_speed - 1) * 100
                    print(f"{result['detector']:<20} {relative_speed:.2f}x {'faster' if speed_change > 0 else 'slower'} ({speed_change:+.1f}%)")
    
    # Save results
    output_file = "timing_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()