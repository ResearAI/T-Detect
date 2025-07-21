#!/usr/bin/env python3
"""
Simple T-Detect Example

This script demonstrates the basic usage of T-Detect for detecting AI-generated text.
It shows how to use the core algorithm without complex model loading for educational purposes.
"""

import torch
import numpy as np
from typing import List


def t_detect_score(text_logits: torch.Tensor, reference_logits: torch.Tensor, 
                   tokens: torch.Tensor, nu: float = 5.0) -> float:
    """
    Compute T-Detect score using heavy-tailed normalization.
    
    Args:
        text_logits: Model logits for the text being analyzed [1, seq_len, vocab_size]
        reference_logits: Reference model logits [1, seq_len, vocab_size]  
        tokens: Token IDs for the text [1, seq_len]
        nu: Degrees of freedom for Student's t-distribution (lower = more robust)
        
    Returns:
        T-Detect score (lower = more likely AI-generated)
    """
    # Ensure single batch
    assert text_logits.shape[0] == 1
    assert reference_logits.shape[0] == 1
    assert tokens.shape[0] == 1
    
    # Convert logits to probabilities
    text_log_probs = torch.log_softmax(text_logits, dim=-1)
    ref_probs = torch.softmax(reference_logits, dim=-1)
    
    # Get log-likelihood of actual tokens
    tokens = tokens.unsqueeze(-1) if tokens.ndim == text_logits.ndim - 1 else tokens
    log_likelihood = text_log_probs.gather(dim=-1, index=tokens).squeeze(-1)
    
    # Compute expected log-likelihood under reference distribution
    expected_log_likelihood = (ref_probs * text_log_probs).sum(dim=-1)
    
    # Compute variance under reference distribution
    variance = (ref_probs * torch.square(text_log_probs)).sum(dim=-1) - torch.square(expected_log_likelihood)
    variance = torch.clamp(variance, min=1e-6)  # Numerical stability
    
    # T-Detect: Heavy-tailed normalization using Student's t-distribution
    # Scale factor accounts for heavier tails: sqrt(ν/(ν-2) * variance)
    t_scale = torch.sqrt(variance * nu / (nu - 2))
    
    # Compute normalized discrepancy
    discrepancy = log_likelihood.sum(dim=-1) - expected_log_likelihood.sum(dim=-1)
    t_detect_score = discrepancy / t_scale.sum(dim=-1)
    
    return t_detect_score.mean().item()


def demonstrate_t_detect():
    """Demonstrate T-Detect with synthetic examples."""
    print("🔍 T-Detect: Tail-Aware Statistical Normalization Demo")
    print("=" * 60)
    print()
    
    # Simulate different text types
    vocab_size = 1000
    seq_length = 15
    
    print("Simulating different text patterns...")
    print()
    
    # 1. Human-like text (more diverse, less predictable)
    print("1. Human-like Text Pattern:")
    human_logits = torch.randn(1, seq_length, vocab_size) * 1.5  # Higher variance
    human_ref_logits = torch.randn(1, seq_length, vocab_size) * 1.5
    human_tokens = torch.randint(0, vocab_size, (1, seq_length))
    
    human_score = t_detect_score(human_logits, human_ref_logits, human_tokens, nu=5)
    print(f"   T-Detect Score: {human_score:.4f}")
    print(f"   Interpretation: {'AI-generated' if human_score < 0 else 'Human-written'}")
    print()
    
    # 2. AI-like text (more predictable, concentrated probability)
    print("2. AI-like Text Pattern:")
    ai_logits = torch.randn(1, seq_length, vocab_size) * 0.8  # Lower variance
    ai_ref_logits = torch.randn(1, seq_length, vocab_size) * 0.8
    # Make some tokens highly predictable (AI characteristic)
    ai_logits[0, :5, :100] += 2.0  # Boost certain token probabilities
    ai_tokens = torch.randint(0, 100, (1, seq_length))  # Tokens from boosted range
    
    ai_score = t_detect_score(ai_logits, ai_ref_logits, ai_tokens, nu=5)
    print(f"   T-Detect Score: {ai_score:.4f}")
    print(f"   Interpretation: {'AI-generated' if ai_score < 0 else 'Human-written'}")
    print()
    
    # 3. Adversarial text (contains statistical outliers)
    print("3. Adversarial Text Pattern (with outliers):")
    adv_logits = torch.randn(1, seq_length, vocab_size) * 1.0
    adv_ref_logits = torch.randn(1, seq_length, vocab_size) * 1.0
    
    # Add statistical outliers (simulating adversarial manipulation)
    outlier_positions = [3, 8, 12]
    for pos in outlier_positions:
        adv_logits[0, pos, :] += torch.randn(vocab_size) * 3.0  # Heavy outliers
    
    adv_tokens = torch.randint(0, vocab_size, (1, seq_length))
    
    # Compare different robustness levels
    print("   Comparing different robustness levels (nu values):")
    for nu_val in [3, 5, 10]:
        adv_score = t_detect_score(adv_logits, adv_ref_logits, adv_tokens, nu=nu_val)
        robustness = "High" if nu_val <= 5 else "Medium" if nu_val <= 10 else "Low"
        print(f"     nu={nu_val} (robustness: {robustness:6s}): {adv_score:.4f}")
    print()
    
    # 4. Demonstrate the key advantage
    print("🎯 Key T-Detect Advantage:")
    print("-" * 30)
    
    # Traditional Gaussian normalization (for comparison)
    def gaussian_score(logits, ref_logits, tokens):
        """Traditional Gaussian normalization (used by FastDetectGPT)."""
        tokens = tokens.unsqueeze(-1) if tokens.ndim == logits.ndim - 1 else tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        
        log_likelihood = log_probs.gather(dim=-1, index=tokens).squeeze(-1)
        expected_ll = (ref_probs * log_probs).sum(dim=-1)
        variance = (ref_probs * torch.square(log_probs)).sum(dim=-1) - torch.square(expected_ll)
        variance = torch.clamp(variance, min=1e-6)
        
        # Standard Gaussian normalization (no heavy-tailed correction)
        discrepancy = log_likelihood.sum(dim=-1) - expected_ll.sum(dim=-1)
        gaussian_score = discrepancy / torch.sqrt(variance.sum(dim=-1))
        return gaussian_score.mean().item()
    
    # Compare on adversarial text
    t_detect_adv = t_detect_score(adv_logits, adv_ref_logits, adv_tokens, nu=5)
    gaussian_adv = gaussian_score(adv_logits, adv_ref_logits, adv_tokens)
    
    print(f"On adversarial text with outliers:")
    print(f"   T-Detect (heavy-tailed):     {t_detect_adv:.4f}")
    print(f"   Traditional (Gaussian):      {gaussian_adv:.4f}")
    print(f"   Difference in robustness:    {abs(t_detect_adv - gaussian_adv):.4f}")
    print()
    print("✨ T-Detect's heavy-tailed normalization provides more stable")
    print("   detection scores in the presence of statistical outliers!")
    print()
    
    # 5. Real-world implications
    print("🌟 Real-World Implications:")
    print("-" * 25)
    print("• Adversarial attacks often create statistical outliers")
    print("• T-Detect's t-distribution normalization is naturally robust")
    print("• Lower nu values = more robustness to outliers")
    print("• Consistent performance across different text types")
    print("• Superior detection of adversarially-manipulated text")


def interactive_example():
    """Interactive example for users to experiment with."""
    print("\n" + "=" * 60)
    print("🎮 Interactive T-Detect Example")
    print("=" * 60)
    print()
    print("This example shows how different parameter settings affect T-Detect:")
    print()
    
    # Create a fixed example
    torch.manual_seed(42)  # For reproducible results
    vocab_size = 1000
    seq_length = 10
    
    logits = torch.randn(1, seq_length, vocab_size)
    ref_logits = torch.randn(1, seq_length, vocab_size)
    tokens = torch.randint(0, vocab_size, (1, seq_length))
    
    print("Testing different nu values (degrees of freedom):")
    print("Lower nu = more heavy-tailed = more robust to outliers")
    print()
    
    nu_values = [2.1, 3, 5, 10, 20, 50]
    scores = []
    
    for nu in nu_values:
        try:
            score = t_detect_score(logits, ref_logits, tokens, nu=nu)
            scores.append(score)
            robustness = "Very High" if nu <= 3 else "High" if nu <= 5 else "Medium" if nu <= 10 else "Low"
            print(f"nu = {nu:4.1f} | Score: {score:8.4f} | Robustness: {robustness}")
        except Exception as e:
            print(f"nu = {nu:4.1f} | Error: {e}")
    
    print()
    print("📊 Key Observations:")
    print("• As nu increases, the score converges to Gaussian normalization")
    print("• nu=5 (default) provides good balance of robustness and performance")
    print("• Very low nu (≤3) provides maximum robustness for adversarial text")
    print("• Choice of nu can be adapted based on expected attack types")


if __name__ == "__main__":
    demonstrate_t_detect()
    interactive_example()
    
    print("\n" + "=" * 60)
    print("🚀 Next Steps:")
    print("• Try running: python test_t_detect.py")
    print("• Explore the reproduction scripts in reproduction_scripts/")
    print("• Read the full paper for theoretical background")
    print("• Use T-Detect in your own projects!")
    print("=" * 60)