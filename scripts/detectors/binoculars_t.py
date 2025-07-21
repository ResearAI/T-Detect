import numpy as np
import torch
import transformers
from .detector_base import DetectorBase
from .model import load_tokenizer, load_model

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:0" if torch.cuda.device_count() > 1 else DEVICE_1

def perplexity_t(encoding: transformers.BatchEncoding,
                 logits: torch.Tensor,
                 median: bool = False,
                 temperature: float = 1.0,
                 nu: float = 5.0):
    """
    Compute perplexity with t-distribution normalization for heavy-tailed distributions.
    """
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ce_losses = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
        # Apply t-distribution scaling for heavy-tailed behavior
        mean_loss = ce_losses.sum(1) / shifted_attention_mask.sum(1)
        var_loss = ((ce_losses - mean_loss.unsqueeze(1)) ** 2 * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        var_loss = torch.clamp(var_loss, min=1e-6)  # Avoid division by zero
        
        # t-distribution scaling: sqrt(nu/(nu-2)) * var
        t_scale = torch.sqrt(nu / (nu - 2) * var_loss)
        ppl = mean_loss / t_scale
        ppl = ppl.to("cpu").float().numpy()

    return ppl

def entropy_t(p_logits: torch.Tensor,
              q_logits: torch.Tensor,
              encoding: transformers.BatchEncoding,
              pad_token_id: int,
              median: bool = False,
              sample_p: bool = False,
              temperature: float = 1.0,
              nu: float = 5.0):
    """
    Compute cross-entropy with t-distribution normalization for heavy-tailed distributions.
    """
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    softmax_fn = torch.nn.Softmax(dim=-1)
    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        # Apply t-distribution scaling for heavy-tailed behavior
        mean_ce = (ce * padding_mask).sum(1) / padding_mask.sum(1)
        var_ce = ((ce - mean_ce.unsqueeze(1)) ** 2 * padding_mask).sum(1) / padding_mask.sum(1)
        var_ce = torch.clamp(var_ce, min=1e-6)  # Avoid division by zero
        
        # t-distribution scaling: sqrt(nu/(nu-2)) * var
        t_scale = torch.sqrt(nu / (nu - 2) * var_ce)
        agg_ce = mean_ce / t_scale
        agg_ce = agg_ce.to("cpu").float().numpy()

    return agg_ce

def compute_text_perplexity(logits: torch.Tensor, 
                           encoding: transformers.BatchEncoding) -> float:
    """
    Compute standard perplexity for dynamic threshold calibration.
    """
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()
    
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    ce_losses = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask
    perplexity = torch.exp(ce_losses.sum() / shifted_attention_mask.sum())
    return perplexity.item()

def assert_tokenizer_consistency(tokenizer1, tokenizer2):
    identical_tokenizers = (
            tokenizer1.vocab == tokenizer2.vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical.")

class BinocularsT(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.observer_model = load_model(self.config.observer_name, self.config.device, self.config.cache_dir)
        self.performer_model = load_model(self.config.performer_name, self.config.device, self.config.cache_dir)
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = load_tokenizer(self.config.observer_name, self.config.cache_dir)
        tokenizer2 = load_tokenizer(self.config.performer_name, self.config.cache_dir)
        assert_tokenizer_consistency(self.tokenizer, tokenizer2)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # t-distribution parameters
        self.nu = getattr(self.config, 'nu', 5.0)  # degrees of freedom
        
        # Dynamic threshold parameters (will be calibrated)
        self.alpha = getattr(self.config, 'alpha', 1.0)
        self.beta = getattr(self.config, 'beta', 0.1)
        self.ref_entropy = getattr(self.config, 'ref_entropy', 5.0)

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.config.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_crit(self, text):
        batch = [text] if isinstance(text, str) else text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        
        # Use t-distribution normalization
        ppl = perplexity_t(encodings, performer_logits, nu=self.nu)
        x_ppl = entropy_t(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                         encodings.to(DEVICE_1), self.tokenizer.pad_token_id, nu=self.nu)
        
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(text, str) else binoculars_scores

    def compute_crit_with_perplexity(self, text):
        """
        Compute binoculars score along with text perplexity for dynamic thresholding.
        """
        batch = [text] if isinstance(text, str) else text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        
        # Compute binoculars score with t-distribution
        ppl = perplexity_t(encodings, performer_logits, nu=self.nu)
        x_ppl = entropy_t(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                         encodings.to(DEVICE_1), self.tokenizer.pad_token_id, nu=self.nu)
        
        binoculars_scores = ppl / x_ppl
        
        # Compute text perplexity for dynamic threshold
        text_perplexity = compute_text_perplexity(performer_logits, encodings)
        
        if isinstance(text, str):
            return binoculars_scores.tolist()[0], text_perplexity
        else:
            return binoculars_scores.tolist(), [text_perplexity] * len(text)

    def get_dynamic_threshold(self, text_perplexity: float) -> float:
        """
        Compute dynamic threshold based on text perplexity.
        """
        entropy = np.log(text_perplexity)
        threshold = self.alpha * np.exp(self.beta * (entropy - self.ref_entropy))
        return threshold

    def calibrate_parameters(self, dev_texts, dev_labels):
        """
        Calibrate alpha, beta, and ref_entropy parameters on development set.
        """
        perplexities = []
        scores = []
        
        for text in dev_texts:
            score, perplexity = self.compute_crit_with_perplexity(text)
            scores.append(score)
            perplexities.append(perplexity)
        
        # Set reference entropy as median of development set
        self.ref_entropy = np.log(np.median(perplexities))
        
        # Grid search for alpha and beta
        best_f1 = 0
        best_alpha = 1.0
        best_beta = 0.1
        
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            for beta in [0.05, 0.1, 0.2, 0.3]:
                predictions = []
                for i, score in enumerate(scores):
                    threshold = alpha * np.exp(beta * (np.log(perplexities[i]) - self.ref_entropy))
                    predictions.append(1 if score > threshold else 0)
                
                # Calculate F1 score
                tp = sum(1 for p, l in zip(predictions, dev_labels) if p == 1 and l == 1)
                fp = sum(1 for p, l in zip(predictions, dev_labels) if p == 1 and l == 0)
                fn = sum(1 for p, l in zip(predictions, dev_labels) if p == 0 and l == 1)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
                    best_beta = beta
        
        self.alpha = best_alpha
        self.beta = best_beta
        print(f"Calibrated parameters: alpha={self.alpha}, beta={self.beta}, ref_entropy={self.ref_entropy}")