import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from data_loader import get_dataloaders
from gan_framework import create_trainer
from evaluation import Evaluator, BenchmarkEvaluator

def run_experiment(dataset_name: str, model_names: list = None, save_history: bool = True):
    """Run experiment on a single dataset with multiple models"""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset_name.upper()}")
    print("="*80)
    
    if model_names is None:
        model_names = config.BENCHMARK_MODELS
    
    # Get dataloaders
    train_loader, val_loader, test_loader, scaler = get_dataloaders(dataset_name)
    
    # Get real test data for evaluation
    real_test_data = []
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            real_test_data.append(batch[0].numpy())
        else:
            real_test_data.append(batch.numpy())
    real_test = np.concatenate(real_test_data, axis=0)
    
    # Get data dimensions
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_data = sample_batch[0]
    else:
        sample_data = sample_batch
    
    n_features = sample_data.shape[2]
    seq_len = sample_data.shape[1]
    
    print(f"\nData dimensions: {n_features} features, {seq_len} timesteps")
    
    # Initialize evaluator
    evaluator = Evaluator(real_test)
    benchmark = BenchmarkEvaluator()
    
    # Create history directory
    if save_history:
        history_dir = os.path.join(config.RESULTS_DIR, 'training_history', dataset_name)
        os.makedirs(history_dir, exist_ok=True)
    
    # Run each model
    results = {}
    synthetic_samples = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Run multiple times for statistical significance
        model_results = []
        model_samples = []
        
        for run in range(config.N_RUNS_PER_MODEL):
            print(f"\nRun {run+1}/{config.N_RUNS_PER_MODEL}")
            
            try:
                # Create trainer
                trainer = create_trainer(model_name, n_features, seq_len, config.DEVICE)
                
                # Create run-specific save directory
                if save_history:
                    run_save_dir = os.path.join(history_dir, f"{model_name}_run_{run+1}")
                else:
                    run_save_dir = None
                
                # Train with history saving
                history = trainer.train(
                    train_loader, 
                    val_loader, 
                    epochs=config.EPOCHS,
                    save_history=save_history,
                    save_dir=run_save_dir
                )
                
                # Generate samples
                samples = trainer.generate(config.N_SYNTHETIC_SAMPLES).numpy()
                model_samples.append(samples)
                
                # Evaluate
                metrics = evaluator.calculate_all_metrics(samples, f"{model_name}_run{run+1}")
                model_results.append(metrics)
                
                print(f"  Composite Score: {metrics.get('Composite_Score', 0):.3f}")
                
                # Save individual run results
                if save_history:
                    run_results = {
                        'model': model_name,
                        'run': run + 1,
                        'metrics': {k: float(v) for k, v in metrics.items()},
                        'history_file': f"{run_save_dir}/final_history.json"
                    }
                    
                    with open(os.path.join(run_save_dir, 'run_results.json'), 'w') as f:
                        json.dump(run_results, f, indent=2)
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        # Average metrics across runs
        if model_results:
            avg_metrics = {}
            for key in model_results[0].keys():
                values = [r[key] for r in model_results if not np.isnan(r[key])]
                if values:
                    avg_metrics[key] = np.mean(values)
            
            results[model_name] = avg_metrics
            synthetic_samples[model_name] = model_samples
            
            # Add to benchmark
            benchmark.add_result(dataset_name, model_name, avg_metrics)
    
    return results, synthetic_samples, benchmark

def run_full_benchmark():
    """Run benchmark across all datasets and models"""
    
    print("\n" + "="*80)
    print("FULL BENCHMARK: 3 DATASETS × 7 GAN MODELS")
    print("="*80)
    
    start_time = time.time()
    
    # Run experiments on each dataset
    for dataset_name in config.DATASET_NAMES:
        results, samples, benchmark = run_experiment(dataset_name)
        
        # Save intermediate results
        save_dir = os.path.join(config.RESULTS_DIR, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save results
     
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            # Convert numpy values to Python types
            serializable_results = {}
            for model, metrics in results.items():
                serializable_results[model] = {k: float(v) for k, v in metrics.items()}
            json.dump(serializable_results, f, indent=2)
        
        # Save synthetic samples
        for model, samples_list in samples.items():
            np.save(os.path.join(save_dir, f'{model}_samples.npy'), samples_list[0])
    
    # Save final benchmark results
    benchmark.save_results(config.RESULTS_DIR)
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETED!")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Results saved to: {config.RESULTS_DIR}/")
    print(f"{'='*80}")

def create_comparison_plot(results: dict, dataset_name: str, save_dir: str = None):
    """Create comparison plot of model performances"""
    
    models = list(results.keys())
    composite_scores = [results[m].get('Composite_Score', 0) for m in models]
    
    # Sort by score
    sorted_indices = np.argsort(composite_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    scores = [composite_scores[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(range(len(models)), scores, color=colors)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Composite Score')
    ax.set_title(f'Model Comparison - {dataset_name}')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    
    plt.show()

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("UNIVARIATE TIME SERIES AUGMENTATION BENCHMARK")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Models: {', '.join(config.BENCHMARK_MODELS)}")
    print(f"Datasets: {', '.join(config.DATASET_NAMES)}")
    print("="*80)
    
    # Ask user for mode
    print("\nSelect mode:")
    print("1. Run full benchmark (all datasets × all models)")
    print("2. Run single dataset experiment")
    print("3. Quick test (1 epoch, 1 run)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_full_benchmark()
    
    elif choice == "2":
        print("\nAvailable datasets:")
        for i, name in enumerate(config.DATASET_NAMES, 1):
            print(f"{i}. {name}")
        
        dataset_idx = int(input("\nSelect dataset number: ")) - 1
        dataset_name = config.DATASET_NAMES[dataset_idx]
        
        results, samples, benchmark = run_experiment(dataset_name)
        
        # Create comparison plot
        create_comparison_plot(results, dataset_name, config.RESULTS_DIR)
        
        # Show best model
        best_model = max(results.items(), key=lambda x: x[1].get('Composite_Score', 0))
        print(f"\nBest model: {best_model[0]} (Score: {best_model[1]['Composite_Score']:.3f})")
    
    elif choice == "3":
        # Quick test
        config.EPOCHS = 1
        config.N_RUNS_PER_MODEL = 1
        
        dataset_name = config.DATASET_NAMES[0]
        model_names = config.BENCHMARK_MODELS[:2]  # First two models
        
        results, samples, benchmark = run_experiment(dataset_name, model_names)
        
        print("\nQuick test complete!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()