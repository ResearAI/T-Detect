# T-Detect: Tail-Aware Statistical Normalization for Robust AI Text Detection
#
# This implementation uses Student's t-distribution normalization to address
# the fundamental statistical flaw in existing curvature-based detectors.
# 
# Key Innovation: Replaces Gaussian normalization with heavy-tailed normalization
# to handle statistical outliers characteristic of adversarial text.
#
# Licensed under the MIT License.
import torch
import numpy as np
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def get_t_discrepancy_analytic(logits_ref, logits_score, labels, nu=5):
    """
    Core T-Detect algorithm: Compute discrepancy using heavy-tailed Student's t-distribution normalization.
    
    This function implements the key innovation of T-Detect by replacing the standard
    Gaussian normalization with a robust normalization based on the Student's t-distribution.
    
    Mathematical formulation:
    T-Detect Score = discrepancy / sqrt((ν/(ν-2)) * variance)
    
    Where:
    - discrepancy = sum(log_likelihood - mean_ref)
    - variance = sum(var_ref) 
    - ν is the degrees of freedom parameter
    
    The factor ν/(ν-2) accounts for the higher variance expected in heavy-tailed data,
    providing superior resilience to statistical outliers characteristic of adversarial text.
    
    Args:
        logits_ref (torch.Tensor): Reference model logits [batch_size, seq_len, vocab_size]
        logits_score (torch.Tensor): Scoring model logits [batch_size, seq_len, vocab_size]
        labels (torch.Tensor): Ground truth token labels [batch_size, seq_len]
        nu (float): Degrees of freedom for t-distribution (default 5 for heavy tails)
                   Lower values = more heavy-tailed, higher robustness to outliers
    
    Returns:
        float: T-Detect discrepancy score. Lower scores indicate higher likelihood of AI generation.
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    
    # Avoid division by zero and negative variances
    var_ref = torch.clamp(var_ref, min=1e-6)
    
    # Compute t-discrepancy with heavy-tailed normalization
    # Scale factor for Student's t-distribution: sqrt(nu/(nu-2) * variance)
    scale = torch.sqrt(var_ref * nu / (nu - 2))
    t_discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / scale.sum(dim=-1)
    t_discrepancy = t_discrepancy.mean()
    
    return t_discrepancy.item()

def compute_perplexity(logits, labels):
    """
    Compute perplexity from logits and labels.
    
    Args:
        logits: Model logits
        labels: Ground truth labels
        
    Returns:
        perplexity: Perplexity value
    """
    lprobs = torch.log_softmax(logits, dim=-1)
    nll = torch.nn.functional.nll_loss(lprobs.view(-1, lprobs.size(-1)), labels.view(-1), reduction='none')
    perplexity = torch.exp(nll.mean())
    return perplexity.item()

class TDetect(DetectorBase):
    def __init__(self, config_name):
        """
        Initialize T-Detect with heavy-tailed statistical normalization.
        
        T-Detect addresses the fundamental statistical flaw in existing curvature-based
        detectors by replacing Gaussian normalization with Student's t-distribution normalization.
        
        Args:
            config_name (str): Configuration name for detector setup
        """
        super().__init__(config_name)
        
        # Core T-Detect algorithm
        self.criterion_fn = get_t_discrepancy_analytic
        
        # Load scoring model (performer in the detection process)
        self.scoring_tokenizer = load_tokenizer(self.config.scoring_model_name, self.config.cache_dir)
        self.scoring_model = load_model(self.config.scoring_model_name, self.config.device, self.config.cache_dir)
        self.scoring_model.eval()
        
        # Load reference model if different from scoring model (observer)
        if self.config.reference_model_name != self.config.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.config.reference_model_name, self.config.cache_dir)
            self.reference_model = load_model(self.config.reference_model_name, self.config.device, self.config.cache_dir)
            self.reference_model.eval()
        
        # T-distribution parameters
        self.nu = getattr(self.config, 'nu', 5)  # Degrees of freedom for heavy-tailed normalization
        
        # Advanced features (experimental)
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.beta = getattr(self.config, 'beta', 0.1)
        self.ref_entropy = getattr(self.config, 'ref_entropy', 5.0)
        self.enable_dynamic_threshold = getattr(self.config, 'enable_dynamic_threshold', False)  # Disabled by default
        
        # Store perplexity for optional dynamic thresholding
        self.last_perplexity = None

    def compute_crit(self, text):
        """
        Compute T-Detect criterion score for input text.
        
        This method tokenizes the input text, computes logits from both reference
        and scoring models, and applies the T-Detect heavy-tailed normalization
        to produce a robust detection score.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: T-Detect score. Lower scores indicate higher likelihood of AI generation.
        """
        tokenized = self.scoring_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                           return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.config.reference_model_name == self.config.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, truncation=True, max_length=self.config.max_token_observed,
                                                     return_tensors="pt", padding=True, return_token_type_ids=False).to(self.config.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            
            # Apply T-Detect's heavy-tailed normalization
            crit = self.criterion_fn(logits_ref, logits_score, labels, nu=self.nu)
            
            # Store perplexity for dynamic thresholding
            if self.enable_dynamic_threshold:
                self.last_perplexity = compute_perplexity(logits_score, labels)
            
        return crit
    
    def get_dynamic_threshold(self, base_threshold):
        """
        Compute dynamic threshold based on perplexity.
        
        Args:
            base_threshold: Base threshold from training
            
        Returns:
            dynamic_threshold: Perplexity-adjusted threshold
        """
        if not self.enable_dynamic_threshold or self.last_perplexity is None:
            return base_threshold
            
        # Convert perplexity to entropy: H(p) = log(perplexity)
        entropy = np.log(self.last_perplexity)
        
        # Dynamic threshold scaling: threshold = alpha * exp(beta * (entropy - ref_entropy))
        scale_factor = self.alpha * np.exp(self.beta * (entropy - self.ref_entropy))
        dynamic_threshold = base_threshold * scale_factor
        
        return dynamic_threshold