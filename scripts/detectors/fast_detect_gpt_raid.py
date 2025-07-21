# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

def get_t_discrepancy_analytic(logits_ref, logits_score, labels, nu=5):
    """
    Compute discrepancy using heavy-tailed Student's t-distribution normalization.
    
    Args:
        logits_ref: Reference model logits
        logits_score: Scoring model logits  
        labels: Ground truth labels
        nu: Degrees of freedom for t-distribution (default 5 for heavy tails)
    
    Returns:
        t_discrepancy: Discrepancy score using t-distribution normalization
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

class FastDetectGPTRaid(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.criterion_fn = get_t_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(self.config.scoring_model_name, self.config.cache_dir)
        self.scoring_model = load_model(self.config.scoring_model_name, self.config.device, self.config.cache_dir)
        self.scoring_model.eval()
        if self.config.reference_model_name != self.config.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(self.config.reference_model_name, self.config.cache_dir)
            self.reference_model = load_model(self.config.reference_model_name, self.config.device, self.config.cache_dir)
            self.reference_model.eval()
        
        # Initialize parameters for dynamic threshold calibration
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.beta = getattr(self.config, 'beta', 0.1)
        self.ref_entropy = getattr(self.config, 'ref_entropy', 5.0)
        self.enable_dynamic_threshold = getattr(self.config, 'enable_dynamic_threshold', True)
        
        # Store perplexity for dynamic thresholding
        self.last_perplexity = None

    def compute_crit(self, text):
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
            
            # Compute t-discrepancy
            crit = self.criterion_fn(logits_ref, logits_score, labels)
            
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