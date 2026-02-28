#!/usr/bin/env python3
"""
Run script for univariate time series augmentation benchmark
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Univariate Time Series Augmentation Benchmark")
    
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset name (jena, usdt, humidity, or all)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['oscillatory_bifurcation_gan', 'vanilla_gan', 'wgan_gp', 
                               'tts_gan', 'sig_wgan', 'sisvae', 'vae_gan'],
                       help='Models to run')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per model')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (1 epoch, 1 run)')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment and check dependencies')
    
    args = parser.parse_args()
    
    if args.setup:
        check_dependencies()
        return
    
    # Update config
    from config import config
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.N_RUNS_PER_MODEL = args.runs
    
    if args.quick:
        config.EPOCHS = 1
        config.N_RUNS_PER_MODEL = 1
    
    # Run experiments
    from main import main
    main()

def check_dependencies():
    """Check and install dependencies"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'scipy', 'scikit-learn',
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
    else:
        print("\nAll dependencies satisfied!")

if __name__ == "__main__":
    main()