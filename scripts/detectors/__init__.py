from .detector_base import DetectorBase
from .baselines import Baselines
from .fast_detect_gpt import FastDetectGPT
from .fast_detect_gpt_raid import FastDetectGPTRaid
from .t_detect import TDetect
from .binoculars import Binoculars
from .binoculars_t import BinocularsT
from .glimpse import Glimpse
from .radar import Radar
from .roberta import RoBERTa
from .detect_llm import LRR


def get_detector(name):
    """
    Factory function to create detector instances.
    
    Args:
        name (str): Detector name
        
    Returns:
        DetectorBase: Initialized detector instance
        
    Available detectors:
        - t_detect: T-Detect with heavy-tailed normalization (recommended)
        - fast_detect: Original FastDetectGPT
        - fast_detect_raid: FastDetectGPT variant
        - binoculars: Cross-perplexity based detector
        - binoculars_t: Binoculars with t-distribution
        - glimpse: OpenAI API-based detector
        - roberta: Fine-tuned RoBERTa classifier
        - radar: Transformer-based classifier
        - log_perplexity: Log-perplexity baseline
        - log_rank: Log-rank baseline
        - lrr: Log-likelihood to log-rank ratio
    """
    name_detectors = {
        # T-Detect (main contribution)
        't_detect': ('t_detect', TDetect),
        
        # Baseline methods
        'fast_detect': ('fast_detect', FastDetectGPT),
        'fast_detect_raid': ('fast_detect_raid', FastDetectGPTRaid),
        'binoculars': ('binoculars', Binoculars),
        'binoculars_t': ('binoculars_t', BinocularsT),
        
        # Neural classifiers
        'roberta': ('roberta', RoBERTa),
        'radar': ('radar', Radar),
        
        # Statistical baselines
        'log_perplexity': ('log_perplexity', Baselines),
        'log_rank': ('log_rank', Baselines),
        'lrr': ('lrr', LRR),
        
        # API-based
        'glimpse': ('glimpse', Glimpse),
    }
    if name in name_detectors:
        config_name, detector_class = name_detectors[name]
        return detector_class(config_name)
    else:
        raise NotImplementedError
