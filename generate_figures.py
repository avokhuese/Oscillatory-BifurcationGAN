#!/usr/bin/env python3
"""
Generate all publication-quality figures for the paper
"""

import sys
import os
import torch
import numpy as np
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import get_dataloaders
from models import create_model
from evaluation import Evaluator
from visualizations import PublicationVisualizer

def load_trained_model(model_type: str, dataset_name: str, checkpoint_path: str):
    """Load a trained model from checkpoint"""
    from gan_framework import create_trainer
    
    # Load data to get dimensions
    train_loader, val_loader, test_loader, scaler = get_dataloaders(dataset_name)
    
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_data = sample_batch[0]
    else:
        sample_data = sample_batch
    
    n_features = sample_data.shape[2]
    seq_len = sample_data.shape[1]
    
    # Create trainer
    trainer = create_trainer(model_type, n_features, seq_len, config.DEVICE)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        trainer.generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'discriminator_state_dict' in checkpoint:
            trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    
    parser.add_argument('--dataset', type=str, default='jena',
                       help='Dataset name (jena, usdt, humidity)')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to benchmark results CSV file')
    parser.add_argument('--output_dir', type=str, default='./paper_figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GENERATING PAPER FIGURES")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"figures_{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(args.dataset)
    
    # Get real test data
    test_data = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            test_data.append(batch[0].numpy())
        else:
            test_data.append(batch.numpy())
    real_test = np.concatenate(test_data, axis=0)
    
    # Load or generate synthetic data
    synthetic_data = {}
    
    # Try to load from results file if provided
    if args.results_file and os.path.exists(args.results_file):
        import pandas as pd
        results_df = pd.read_csv(args.results_file)
        
        # For each model, generate samples
        for model_name in results_df['Model'].unique():
            print(f"Generating samples for {model_name}...")
            
            # Try to load model checkpoint if available
            if args.model_checkpoint and os.path.exists(args.model_checkpoint):
                trainer = load_trained_model(model_name, args.dataset, args.model_checkpoint)
                if trainer:
                    with torch.no_grad():
                        z = torch.randn(500, config.LATENT_DIM, device=config.DEVICE)
                        samples = trainer.generator(z).cpu().numpy()
                    synthetic_data[model_name] = samples
            else:
                # Generate random samples for demonstration
                print(f"  Using random samples (no checkpoint provided)")
                synthetic_data[model_name] = np.random.randn(500, real_test.shape[1], 1) * 0.5
    
    # If no models loaded, create placeholder synthetic data
    if not synthetic_data:
        print("\nNo models loaded. Creating placeholder synthetic data...")
        model_names = ['Oscillatory_BifurcationGAN', 'Vanilla_GAN', 'WGAN_GP', 'TTS_GAN']
        for model_name in model_names:
            synthetic_data[model_name] = np.random.randn(500, real_test.shape[1], 1) * 0.5
    
    # Create visualizer
    visualizer = PublicationVisualizer(save_dir=save_dir)
    
    # Load benchmark results if available
    benchmark_results = None
    if args.results_file and os.path.exists(args.results_file):
        benchmark_results = pd.read_csv(args.results_file)
    
    # Load model for dynamics visualization
    generator = None
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        trainer = load_trained_model('oscillatory_bifurcation_gan', args.dataset, args.model_checkpoint)
        if trainer:
            generator = trainer.generator
    
    # Generate all figures
    figures = visualizer.generate_all_figures(
        real_data=real_test,
        synthetic_data=synthetic_data,
        benchmark_results=benchmark_results,
        generator=generator,
        device=config.DEVICE,
        dataset_name=args.dataset
    )
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"Figures saved to: {save_dir}")
    
    # List generated figures
    print("\nGenerated figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")
    
    print("="*80)

if __name__ == "__main__":
    main()