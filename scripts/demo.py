#!/usr/bin/env python3
"""
T-Detect Demo Script

This script provides various demonstration modes for T-Detect:
- Interactive text detection
- Batch processing
- Visualization of detection scores
- Comparison with baseline methods
- Statistical analysis of score distributions

Usage:
    python scripts/demo.py --interactive
    python scripts/demo.py --text "Your text here"
    python scripts/demo.py --visualize --dataset raid.test
    python scripts/demo.py --compare --methods t_detect,fast_detect,binoculars
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from detectors import get_detector
import utils


class TDetectDemo:
    """T-Detect demonstration interface."""
    
    def __init__(self):
        """Initialize demo with available detectors."""
        self.available_detectors = [
            't_detect', 'fast_detect', 'binoculars', 'log_perplexity', 'roberta'
        ]
        
    def interactive_mode(self):
        """Run interactive text detection mode."""
        print("=" * 60)
        print("T-DETECT INTERACTIVE DEMO")
        print("=" * 60)
        print()
        print("T-Detect uses heavy-tailed statistical normalization to detect")
        print("AI-generated text with superior robustness to adversarial attacks.")
        print()
        print("Enter 'quit' to exit, 'help' for commands, or paste text to analyze.")
        print()
        
        # Initialize detector
        print("Loading T-Detect (this may take a moment)...")
        try:
            detector = get_detector('t_detect')
            print("✓ T-Detect loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading T-Detect: {e}")
            return
        print()
        
        while True:
            text = input("📝 Enter text to analyze: ").strip()
            
            if text.lower() == 'quit':
                print("Goodbye!")
                break
            elif text.lower() == 'help':
                self._show_help()
                continue
            elif not text:
                continue
                
            # Analyze text
            print("\n🔍 Analyzing...")
            try:
                score = detector.compute_crit(text)
                
                # Simple threshold for demo (would be trained in practice)
                threshold = 0.0
                is_ai = score > threshold
                confidence = abs(score)
                
                print(f"\n📊 RESULTS:")
                print(f"   T-Detect Score: {score:.4f}")
                print(f"   Prediction: {'🤖 AI-generated' if is_ai else '👤 Human-written'}")
                print(f"   Confidence: {confidence:.4f}")
                
                # Additional analysis
                if confidence > 2.0:
                    conf_level = "Very High"
                elif confidence > 1.0:
                    conf_level = "High"
                elif confidence > 0.5:
                    conf_level = "Medium"
                else:
                    conf_level = "Low"
                    
                print(f"   Confidence Level: {conf_level}")
                
                # Statistical insights
                if abs(score) > 2.0:
                    print("   ⚠️  Statistical outlier detected - T-Detect's heavy-tailed")
                    print("       normalization provides robust handling of this case.")
                
            except Exception as e:
                print(f"✗ Error during analysis: {e}")
            
            print()
    
    def analyze_text(self, text: str, detector_name: str = 't_detect'):
        """Analyze a single text with specified detector."""
        print(f"Analyzing text with {detector_name}...")
        
        try:
            detector = get_detector(detector_name)
            score = detector.compute_crit(text)
            
            print(f"\nText: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
            print(f"Detector: {detector_name}")
            print(f"Score: {score:.4f}")
            print(f"Prediction: {'AI-generated' if score > 0 else 'Human-written'}")
            
            return score
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def compare_methods(self, text: str, methods: List[str]):
        """Compare multiple detection methods on the same text."""
        print("=" * 60)
        print("MULTI-METHOD COMPARISON")
        print("=" * 60)
        print()
        print(f"Text: \"{text[:150]}{'...' if len(text) > 150 else ''}\"")
        print()
        
        results = {}
        
        for method in methods:
            print(f"🔍 Running {method}...")
            try:
                start_time = time.time()
                detector = get_detector(method)
                score = detector.compute_crit(text)
                duration = time.time() - start_time
                
                results[method] = {
                    'score': score,
                    'prediction': 'AI-generated' if score > 0 else 'Human-written',
                    'duration': duration
                }
                
                print(f"   Score: {score:.4f}")
                print(f"   Prediction: {results[method]['prediction']}")
                print(f"   Time: {duration:.2f}s")
                
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results[method] = None
            print()
        
        # Summary comparison
        print("📊 COMPARISON SUMMARY:")
        print("-" * 30)
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if valid_results:
            # Show score rankings
            sorted_by_score = sorted(valid_results.items(), key=lambda x: x[1]['score'])
            
            print("Score Rankings (lower = more likely AI):")
            for i, (method, result) in enumerate(sorted_by_score, 1):
                print(f"  {i}. {method}: {result['score']:.4f}")
            
            print()
            
            # Show speed comparison
            sorted_by_speed = sorted(valid_results.items(), key=lambda x: x[1]['duration'])
            print("Speed Rankings (faster = better):")
            for i, (method, result) in enumerate(sorted_by_speed, 1):
                print(f"  {i}. {method}: {result['duration']:.2f}s")
            
            # Highlight T-Detect advantages
            if 't_detect' in valid_results:
                print()
                print("✨ T-Detect Advantages:")
                print("   • Heavy-tailed statistical normalization")
                print("   • Superior robustness to adversarial attacks")
                print("   • Stable performance across text types")
                
        return results
    
    def visualize_distributions(self, dataset_path: str, methods: List[str] = None):
        """Visualize detection score distributions."""
        if methods is None:
            methods = ['t_detect', 'fast_detect', 'binoculars']
            
        print(f"📈 Visualizing score distributions for dataset: {dataset_path}")
        
        # Load dataset
        try:
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        # Sample subset for visualization (to speed up demo)
        if len(data) > 100:
            import random
            data = random.sample(data, 100)
            print(f"Using random sample of 100 items for visualization")
        
        fig, axes = plt.subplots(len(methods), 1, figsize=(10, 4*len(methods)))
        if len(methods) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods):
            print(f"Computing scores with {method}...")
            
            try:
                detector = get_detector(method)
                human_scores = []
                ai_scores = []
                
                for item in data:
                    text = item.get('generation', item.get('text', ''))
                    if not text:
                        continue
                        
                    score = detector.compute_crit(text)
                    
                    # Assume 'source' field indicates human (0) vs AI (1)
                    source = item.get('source', item.get('label', 0))
                    if source == 0:
                        human_scores.append(score)
                    else:
                        ai_scores.append(score)
                
                # Plot distributions
                ax = axes[i]
                ax.hist(human_scores, bins=30, alpha=0.7, label='Human', color='blue')
                ax.hist(ai_scores, bins=30, alpha=0.7, label='AI', color='red')
                ax.set_xlabel('Detection Score')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{method} Score Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistical info
                if human_scores and ai_scores:
                    human_mean = np.mean(human_scores)
                    ai_mean = np.mean(ai_scores)
                    separation = abs(human_mean - ai_mean)
                    ax.axvline(human_mean, color='blue', linestyle='--', alpha=0.8)
                    ax.axvline(ai_mean, color='red', linestyle='--', alpha=0.8)
                    ax.text(0.02, 0.98, f'Separation: {separation:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error with {method}: {e}")
                axes[i].text(0.5, 0.5, f"Error: {e}", transform=axes[i].transAxes,
                           ha='center', va='center')
        
        plt.tight_layout()
        
        # Save plot
        output_path = f"score_distributions_{len(methods)}methods.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_path}")
        plt.show()
    
    def _show_help(self):
        """Show help information."""
        print()
        print("COMMANDS:")
        print("  help  - Show this help message")
        print("  quit  - Exit the demo")
        print()
        print("ABOUT T-DETECT:")
        print("  T-Detect improves upon existing detectors by using")
        print("  Student's t-distribution normalization instead of")
        print("  standard Gaussian normalization. This provides:")
        print()
        print("  ✓ Superior robustness to adversarial attacks")
        print("  ✓ Better handling of statistical outliers")
        print("  ✓ More stable performance across text types")
        print("  ✓ Heavy-tailed statistical foundation")
        print()
        print("SCORE INTERPRETATION:")
        print("  Lower scores → More likely AI-generated")
        print("  Higher scores → More likely Human-written")
        print("  Score magnitude → Confidence level")
        print()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="T-Detect Demo Script")
    parser.add_argument('--interactive', action='store_true',
                       help="Run interactive demo mode")
    parser.add_argument('--text', type=str,
                       help="Analyze specific text")
    parser.add_argument('--detector', type=str, default='t_detect',
                       help="Detector to use (default: t_detect)")
    parser.add_argument('--compare', action='store_true',
                       help="Compare multiple methods")
    parser.add_argument('--methods', type=str, default='t_detect,fast_detect,binoculars',
                       help="Comma-separated list of methods to compare")
    parser.add_argument('--visualize', action='store_true',
                       help="Visualize score distributions")
    parser.add_argument('--dataset', type=str,
                       help="Dataset file for visualization")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device to use (cuda/cpu)")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = TDetectDemo()
    
    # Set environment variables
    os.environ['DEVICE'] = args.device
    
    if args.interactive:
        demo.interactive_mode()
        
    elif args.text:
        if args.compare:
            methods = args.methods.split(',')
            demo.compare_methods(args.text, methods)
        else:
            demo.analyze_text(args.text, args.detector)
            
    elif args.visualize and args.dataset:
        methods = args.methods.split(',') if args.compare else ['t_detect']
        
        # Find dataset file
        dataset_paths = [
            args.dataset,
            f"benchmark/hart/{args.dataset}",
            f"benchmark/raid/{args.dataset}"
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
                
        if dataset_path:
            demo.visualize_distributions(dataset_path, methods)
        else:
            print(f"Dataset not found: {args.dataset}")
            print(f"Searched paths: {dataset_paths}")
            
    else:
        # Default: show example usage
        print("T-Detect Demo - Example Usage:")
        print()
        print("Interactive mode:")
        print("  python scripts/demo.py --interactive")
        print()
        print("Analyze specific text:")
        print("  python scripts/demo.py --text \"Your text here\"")
        print()
        print("Compare methods:")
        print("  python scripts/demo.py --compare --text \"Your text here\" --methods t_detect,fast_detect")
        print()
        print("Visualize distributions:")
        print("  python scripts/demo.py --visualize --dataset raid.test --methods t_detect,binoculars")
        print()
        
        # Quick demo
        print("Quick Demo:")
        print("-" * 20)
        demo_text = ("The rapid advancement of artificial intelligence has fundamentally "
                    "transformed numerous sectors, from healthcare to autonomous systems, "
                    "while simultaneously raising important questions about ethical implications.")
        
        print(f"Sample text: \"{demo_text}\"")
        print()
        demo.analyze_text(demo_text)


if __name__ == "__main__":
    main()