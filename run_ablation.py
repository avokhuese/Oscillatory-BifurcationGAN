#!/usr/bin/env python3
"""
Run ablation study for Oscillatory BifurcationGAN on all datasets
"""

import sys
import os
import torch
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from data_loader import get_dataloaders
from ablation_study import AblationStudy
from visualizations import PublicationVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for Oscillatory BifurcationGAN")
    
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['jena', 'usdt', 'humidity'],
                       help='Dataset names (jena, usdt, humidity) - can specify multiple')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs per configuration')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Output directory for results')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip datasets that already have results')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OSCILLATORY BIFURCATIONGAN ABLATION STUDY")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Epochs per configuration: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Run ablation for each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Create dataset-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_save_dir = os.path.join(args.output_dir, f"ablation_{dataset_name}_{timestamp}")
        os.makedirs(dataset_save_dir, exist_ok=True)
        
        # Check if results already exist
        if args.skip_existing:
            existing_file = os.path.join(args.output_dir, f"ablation_{dataset_name}_*/ablation_results.json")
            import glob
            if glob.glob(existing_file):
                print(f"Results already exist for {dataset_name}. Skipping...")
                continue
        
        # Load data
        print(f"\nLoading {dataset_name} dataset...")
        try:
            train_loader, val_loader, test_loader, scaler = get_dataloaders(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            print("Skipping to next dataset...")
            continue
        
        # Get data dimensions
        try:
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_data = sample_batch[0]
            else:
                sample_data = sample_batch
            
            n_features = sample_data.shape[2]
            seq_len = sample_data.shape[1]
            
            print(f"Data dimensions: {n_features} features, {seq_len} timesteps")
        except Exception as e:
            print(f"Error getting data dimensions: {e}")
            print("Skipping to next dataset...")
            continue
        
        # Run ablation study
        ablation = AblationStudy(
            dataset_name=dataset_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            n_features=n_features,
            seq_len=seq_len,
            device=config.DEVICE,
            save_dir=dataset_save_dir
        )
        
        try:
            results = ablation.run_component_ablation(epochs=args.epochs)
            all_results[dataset_name] = results
            
            # Generate ablation plots
            ablation.plot_ablation_results()
            
            # Generate component analysis table
            table_df = ablation.generate_component_analysis_table()
            
            print(f"\n✓ Ablation study completed for {dataset_name}")
            print(f"  Results saved to: {dataset_save_dir}")
            
        except Exception as e:
            print(f"Error during ablation study for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping to next dataset...")
            continue
    
    # Print overall summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE - OVERALL SUMMARY")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print("-"*60)
        
        # Sort results by composite score
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]['metrics']['Composite_Score'], 
            reverse=True
        )
        
        for name, data in sorted_results[:3]:  # Top 3 configurations
            score = data['metrics']['Composite_Score']
            std = data['metrics'].get('Composite_Score_std', 0)
            print(f"  {name[:40]:40s}: {score:.4f} ± {std:.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()