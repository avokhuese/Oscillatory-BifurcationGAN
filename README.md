# Oscillatory BifurcationGAN: Advanced Time Series Augmentation Framework

## Title
**Oscillatory Bifurcation Generative Adversarial Network (O-BGAN) for Multivariate Time Series Augmentation**

## Description

This repository contains a comprehensive framework for time series data augmentation using novel Generative Adversarial Network architectures. The core innovation is the **Oscillatory BifurcationGAN**, which combines dynamical systems theory with deep learning to generate high-quality synthetic time series data. The framework implements 7 state-of-the-art GAN variants for comprehensive benchmarking:

1. **Oscillatory BifurcationGAN** (Our novel architecture)
2. Vanilla GAN
3. WGAN-GP
4. TTS-GAN (Time Series Synthesis GAN)
5. SigWGAN (Signature WGAN)
6. sisVAE (Simplified Variational Autoencoder)
7. VAE-GAN

The models are evaluated on 3 diverse real-world datasets:
- **Jena Climate Data**: Temperature time series with seasonal patterns
- **USDT Cryptocurrency Data**: Financial time series with 5 features (OHLCV)
- **Humidity Sensor Data**: Environmental sensor readings with daily cycles

## Dataset Information

### 1. Jena Climate Dataset
- **Source**: Max Planck Institute for Biogeochemistry
- **Description**: Daily minimum temperatures recorded in Jena, Germany
- **Characteristics**: Univariate time series with strong seasonal patterns
- **Length**: 5,000+ time points
- **Preprocessing**: StandardScaler normalization

### 2. USDT Cryptocurrency Dataset
- **Source**: Cryptocurrency exchange data
- **Description**: USDT-USD trading data with OHLCV (Open, High, Low, Close, Volume)
- **Characteristics**: Multivariate (5 features), stablecoin with small variations
- **Length**: Variable (synthetic fallback available)
- **Preprocessing**: MinMaxScaler normalization

### 3. Humidity Sensor Dataset
- **Source**: Environmental sensor network
- **Description**: Relative humidity readings from plant sensors
- **Characteristics**: Univariate, daily cycles with random events
- **Length**: 10,000+ time points
- **Preprocessing**: StandardScaler normalization

## Code Information

### Directory Structure

O-BGAN2.0/
- ├── config.py # Configuration parameters
- ├── data_loader.py # Data loading and preprocessing
- ├── models.py # All GAN model architectures
- ├── gan_framework.py # Training framework with history tracking
- ├── evaluation.py # Comprehensive evaluation metrics
- ├── training_dynamics.py # Training dynamics visualization
- ├── ablation_study.py # Component ablation analysis
- ├── visualizations.py # Publication-quality figure generation
- ├── main.py # Main experiment pipeline
- ├── run.py # Entry point script
- ├── run_ablation.py # Ablation study runner
- ├── run_training_analysis.py # Training dynamics analysis
- ├── generate_figures.py # Paper figure generation
- ├── requirements.txt # Dependencies
- └── README.md # This file


### Key Components

#### `config.py`
Central configuration management with dataclasses. Controls all hyperparameters including:
- Model architectures (hidden dimensions, layers)
- Training parameters (learning rates, batch sizes)
- Bifurcation dynamics (Hopf parameters, oscillator coupling)
- Evaluation metrics (FID, MMD, Wasserstein, etc.)

#### `models.py`
Implements 7 GAN architectures:
- **OscillatoryBifurcationGenerator**: Novel architecture with coupled oscillators and Hopf bifurcation dynamics
- **VanillaGenerator**: Standard MLP-based generator
- **WGANGenerator**: WGAN-GP generator with gradient penalty
- **TTSGenerator**: LSTM-based time series generator
- **SigWGANGenerator**: Signature method-based generator
- **sisVAE**: Simplified variational autoencoder
- **VAEGANGenerator**: Hybrid VAE-GAN architecture

#### `gan_framework.py`
Training framework with:
- Mixed precision training (AMP)
- Gradient penalty computation
- History tracking (losses, gradient norms)
- Checkpoint saving/loading
- Multi-seed training support

#### `evaluation.py`
Comprehensive evaluation metrics:
- **Distribution metrics**: JS Divergence, KS Statistic, Wasserstein Distance
- **Temporal metrics**: ACF Similarity, PSD Similarity
- **Quality metrics**: FID, MMD
- **Composite score**: Weighted combination of all metrics

#### `training_dynamics.py`
Training analysis tools:
- Loss curves with confidence intervals
- Gradient norm evolution
- Failure rate analysis
- Convergence speed comparison
- Stability heatmaps

#### `visualizations.py`
Publication-quality figure generation:
- Time series comparison plots
- Distribution analysis (histograms + KDE)
- Temporal dynamics (ACF, PSD, phase space)
- Model comparison bar charts
- Radar charts for multi-metric comparison
- t-SNE manifold visualization

## Usage Instructions

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/O-BGAN2.0.git
cd O-BGAN2.0

2. **Create virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Quick start**
python run.py --quick

5. **Full benchmark**
python run.py

6. **Ablation Study**
python run_ablation.py

7. **Training Dynamics Analysis**
python run_training_analysis.py


8. **Generate Figures**
python generate_figures.py

9. # Acknowledgments
- Max Planck Institute for Biogeochemistry for the Jena Climate Data

- Cryptocurrency exchanges for providing USDT trading data

- The open-source community for PyTorch, scikit-learn, and other libraries
# Citation
@article{avokhuese2026,
  title={Oscillatory Bifurcation Generative Adversarial Networks for Time Series Augmentation},
  author={Alexander Okhuese Victor},
  journal={arXiv preprint},
  year={2026}
}

# Contact
For questions or collaborations

- Email: avokhuese@gmail.com or alexander.victor4@mail.dcu.ie

- Github: @avokhuese

