import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class Config:
    """Configuration for univariate time series augmentation"""
    
    # Dataset parameters
    DATASET_NAMES: List[str] = field(default_factory=lambda: ['jena', 'usdt', 'humidity'])
    
    # Model architecture
    SEQ_LEN: int = 24
    LATENT_DIM: int = 128
    GENERATOR_HIDDEN: int = 256
    DISCRIMINATOR_HIDDEN: int = 256
    NUM_LAYERS: int = 3
    
    # === BIFURCATION GAN PARAMETERS ===
    USE_BIFURCATION: bool = True
    BIFURCATION_TYPE: str = "hopf"  # "hopf", "pitchfork", "saddle_node"
    HOPF_MU: float = 0.1
    HOPF_OMEGA: float = 2.0
    HOPF_ALPHA: float = 0.1
    HOPF_BETA: float = 1.0
    
    # === OSCILLATORY BIFURCATION GAN PARAMETERS ===
    USE_OSCILLATORY_DYNAMICS: bool = True
    N_OSCILLATORS: int = 3
    OSCILLATOR_COUPLING: float = 0.2
    NATURAL_FREQUENCIES: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    PHASE_NOISE_STD: float = 0.1
    AMPLITUDE_DECAY: float = 0.05
    AMPLITUDE_SATURATION: float = 1.0
    
    # === VAE PARAMETERS (for sisVAE and VAE-GAN) ===
    VAE_HIDDEN_DIM: int = 128
    VAE_LATENT_DIM: int = 32
    KL_WEIGHT: float = 0.1
    RECONSTRUCTION_WEIGHT: float = 1.0
    
    # === TRAINING PARAMETERS ===
    BATCH_SIZE: int = 64
    EPOCHS: int = 500
    GENERATOR_LR: float = 2e-4
    DISCRIMINATOR_LR: float = 2e-4
    CRITIC_ITERATIONS: int = 5
    
    # Optimizer
    BETA1: float = 0.5
    BETA2: float = 0.999
    WEIGHT_DECAY: float = 1e-5
    
    # Gradient penalties
    LAMBDA_GP: float = 10.0
    GRADIENT_PENALTY_TYPE: str = "wgan-gp"
    
    # Spectral normalization
    USE_SPECTRAL_NORM: bool = True
    
    # === REGULARIZATION ===
    USE_GRADIENT_PENALTY: bool = True
    USE_CONSISTENCY_REGULARIZATION: bool = True
    CONSISTENCY_LAMBDA: float = 10.0
    USE_FEATURE_MATCHING: bool = True
    FEATURE_MATCHING_LAMBDA: float = 0.1
    DIVERSITY_WEIGHT: float = 0.2
    
    # === EVALUATION METRICS ===
    CALCULATE_FID: bool = True
    CALCULATE_MMD: bool = True
    CALCULATE_WASSERSTEIN: bool = True
    CALCULATE_JSD: bool = True
    CALCULATE_KS_TEST: bool = True
    CALCULATE_ACF_SIMILARITY: bool = True
    CALCULATE_PSD_SIMILARITY: bool = True
    
    # === BENCHMARKING ===
    BENCHMARK_MODELS: List[str] = field(default_factory=lambda: [
        "vanilla_gan", "wgan_gp", "tts_gan", "sig_wgan", 
        "sisvae", "vae_gan", "oscillatory_bifurcation_gan"
    ])
    N_RUNS_PER_MODEL: int = 3
    CONFIDENCE_LEVEL: float = 0.95
    
    # === EXPERIMENTAL SETTINGS ===
    USE_EARLY_STOPPING: bool = True
    PATIENCE: int = 20
    MIN_DELTA: float = 0.001
    
    # === DATA SPLITS ===
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    # === RESOURCE MANAGEMENT ===
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    USE_AMP: bool = True
    
    # === PATHS ===
    DATA_DIR: str = "./data"
    SAVE_DIR: str = "./saved_models"
    RESULTS_DIR: str = "./results"
    LOGS_DIR: str = "./logs"
    CACHE_DIR: str = "./cache"
    
    # === SYNTHETIC DATA ===
    N_SYNTHETIC_SAMPLES: int = 1000
    SYNTHETIC_LENGTH_VARIATION: bool = True
    MIN_SYNTHETIC_LENGTH: int = 50
    MAX_SYNTHETIC_LENGTH: int = 500
    
    # === MODEL SAVING ===
    SAVE_CHECKPOINT_FREQ: int = 10
    SAVE_BEST_MODEL: bool = True
    
    # === RANDOM SEED ===
    SEED: int = 42
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Set random seeds
        torch.manual_seed(self.SEED)
        np.random.seed(self.SEED)
        
        # Create directories
        import os
        directories = [self.DATA_DIR, self.SAVE_DIR, self.RESULTS_DIR, 
                      self.LOGS_DIR, self.CACHE_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters"""
        assert self.LATENT_DIM > 0, "LATENT_DIM must be positive"
        assert self.GENERATOR_HIDDEN > 0, "GENERATOR_HIDDEN must be positive"
        assert self.DISCRIMINATOR_HIDDEN > 0, "DISCRIMINATOR_HIDDEN must be positive"
        assert self.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert 0 < self.TRAIN_SPLIT < 1, "TRAIN_SPLIT must be between 0 and 1"
        assert 0 < self.VAL_SPLIT < 1, "VAL_SPLIT must be between 0 and 1"
        assert self.EPOCHS > 0, "EPOCHS must be positive"
        assert self.N_OSCILLATORS > 0, "N_OSCILLATORS must be positive"
        assert self.BIFURCATION_TYPE in ["hopf", "pitchfork", "saddle_node"], \
            f"Invalid BIFURCATION_TYPE: {self.BIFURCATION_TYPE}"

# Global configuration instance
config = Config()