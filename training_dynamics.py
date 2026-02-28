"""
Training Dynamics Visualization Module
Captures loss curves, stability metrics, gradient norms, and sample quality over time
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class TrainingDynamicsVisualizer:
    """
    Comprehensive training dynamics analysis for GAN stability
    """
    
    def __init__(self, save_dir: str = './training_dynamics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_loss_curves(self, histories: Dict[str, List[Dict]], 
                         model_names: List[str],
                         display_names: List[str],
                         model_colors: List[str],
                         dataset_name: str,
                         num_seeds: int = 5,
                         smooth_sigma: float = 1.0):
        """
        Figure 1: Loss curves with confidence intervals across multiple seeds
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Track if any data was plotted
        has_data = False
        
        for idx, (model_key, display_name, color) in enumerate(zip(model_names, display_names, model_colors)):
            if model_key not in histories:
                print(f"  Warning: {model_key} not in histories")
                continue
            
            model_histories = histories[model_key]
            if not model_histories:
                print(f"  Warning: No histories for {model_key}")
                continue
            
            # Extract metrics across seeds
            g_losses = []
            d_losses = []
            wassersteins = []
            epochs = None
            
            for seed_idx, seed_history in enumerate(model_histories):
                if seed_history and isinstance(seed_history, dict):
                    # Get losses, handling different possible keys
                    g_loss = seed_history.get('g_loss', [])
                    d_loss = seed_history.get('d_loss', [])
                    w_dist = seed_history.get('wasserstein', [])
                    
                    if len(g_loss) > 0 and len(d_loss) > 0:
                        g_losses.append(g_loss)
                        d_losses.append(d_loss)
                        wassersteins.append(w_dist if len(w_dist) > 0 else [0] * len(g_loss))
                        
                        if epochs is None:
                            epochs = list(range(len(g_loss)))
            
            if not g_losses or epochs is None:
                print(f"  Warning: No valid loss data for {model_key}")
                continue
            
            has_data = True
            
            # Convert to numpy arrays with consistent length
            min_len = min([len(g) for g in g_losses])
            g_losses = np.array([g[:min_len] for g in g_losses])
            d_losses = np.array([d[:min_len] for d in d_losses])
            wassersteins = np.array([w[:min_len] for w in wassersteins])
            epochs = np.array(epochs[:min_len])
            
            # Compute statistics
            g_mean = np.mean(g_losses, axis=0)
            g_std = np.std(g_losses, axis=0)
            g_sem = g_std / np.sqrt(len(g_losses))
            
            d_mean = np.mean(d_losses, axis=0)
            d_std = np.std(d_losses, axis=0)
            d_sem = d_std / np.sqrt(len(d_losses))
            
            w_mean = np.mean(wassersteins, axis=0)
            w_std = np.std(wassersteins, axis=0)
            w_sem = w_std / np.sqrt(len(wassersteins))
            
            # Apply Gaussian smoothing
            if smooth_sigma > 0 and len(g_mean) > 10:
                from scipy.ndimage import gaussian_filter1d
                g_mean = gaussian_filter1d(g_mean, sigma=smooth_sigma)
                d_mean = gaussian_filter1d(d_mean, sigma=smooth_sigma)
                w_mean = gaussian_filter1d(w_mean, sigma=smooth_sigma)
            
            # Plot Generator Loss
            ax = axes[0, 0]
            line = ax.plot(epochs, g_mean, color=color, linewidth=2, label=display_name)[0]
            ax.fill_between(epochs, g_mean - g_sem, g_mean + g_sem, 
                           color=color, alpha=0.2)
            
            # Plot Discriminator Loss
            ax = axes[0, 1]
            ax.plot(epochs, d_mean, color=color, linewidth=2, label=display_name)
            ax.fill_between(epochs, d_mean - d_sem, d_mean + d_sem, 
                           color=color, alpha=0.2)
            
            # Plot Wasserstein Distance
            ax = axes[1, 0]
            ax.plot(epochs, w_mean, color=color, linewidth=2, label=display_name)
            ax.fill_between(epochs, w_mean - w_sem, w_mean + w_sem, 
                           color=color, alpha=0.2)
            
            # Plot G/D ratio (training balance)
            ax = axes[1, 1]
            gd_ratio = g_mean / (d_mean + 1e-8)
            ax.plot(epochs, gd_ratio, color=color, linewidth=2, label=display_name)
            lower = (g_mean - g_sem) / (d_mean + d_sem + 1e-8)
            upper = (g_mean + g_sem) / (d_mean - d_sem + 1e-8)
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.2)
        
        if not has_data:
            print(f"  Warning: No data to plot for {dataset_name}")
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No training data available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Format plots
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Generator Loss')
        axes[0, 0].set_title('Generator Loss Convergence', fontweight='bold')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Discriminator Loss')
        axes[0, 1].set_title('Discriminator Loss Convergence', fontweight='bold')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Wasserstein Distance')
        axes[1, 0].set_title('Wasserstein Distance', fontweight='bold')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('G/D Ratio')
        axes[1, 1].set_title('Generator/Discriminator Balance', fontweight='bold')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.suptitle(f'Training Dynamics - {dataset_name} (averaged over {num_seeds} seeds)', 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'loss_curves_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Loss curves saved to: {save_path}")
        return save_path
    
    def plot_gradient_norms(self, gradient_histories: Dict[str, List[Dict]],
                           model_names: List[str],
                           display_names: List[str],
                           model_colors: List[str],
                           dataset_name: str,
                           num_seeds: int = 5):
        """
        Figure 2: Gradient norm evolution during training
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        has_data = False
        
        for idx, (model_key, display_name, color) in enumerate(zip(model_names, display_names, model_colors)):
            if model_key not in gradient_histories:
                continue
            
            grad_data = gradient_histories[model_key]
            if not grad_data:
                continue
            
            # Extract generator and discriminator gradient norms
            g_grads = []
            d_grads = []
            epochs = None
            
            for seed_grads in grad_data:
                if seed_grads and isinstance(seed_grads, dict):
                    g_grad = seed_grads.get('g_grad_norm', [])
                    d_grad = seed_grads.get('d_grad_norm', [])
                    
                    if len(g_grad) > 0 and len(d_grad) > 0:
                        g_grads.append(g_grad)
                        d_grads.append(d_grad)
                        
                        if epochs is None:
                            epochs = list(range(len(g_grad)))
            
            if not g_grads or epochs is None:
                continue
            
            has_data = True
            
            # Convert to arrays
            min_len = min([len(g) for g in g_grads])
            g_grads = np.array([g[:min_len] for g in g_grads])
            d_grads = np.array([d[:min_len] for d in d_grads])
            epochs = np.array(epochs[:min_len])
            
            # Compute statistics
            g_mean = np.mean(g_grads, axis=0)
            g_std = np.std(g_grads, axis=0)
            g_sem = g_std / np.sqrt(len(g_grads))
            
            d_mean = np.mean(d_grads, axis=0)
            d_std = np.std(d_grads, axis=0)
            d_sem = d_std / np.sqrt(len(d_grads))
            
            # Plot Generator Gradients
            ax = axes[0, 0]
            ax.plot(epochs, g_mean, color=color, linewidth=2, label=display_name)
            ax.fill_between(epochs, g_mean - g_sem, g_mean + g_sem, 
                           color=color, alpha=0.2)
            
            # Plot Discriminator Gradients
            ax = axes[0, 1]
            ax.plot(epochs, d_mean, color=color, linewidth=2, label=display_name)
            ax.fill_between(epochs, d_mean - d_sem, d_mean + d_sem, 
                           color=color, alpha=0.2)
            
            # Plot Gradient Ratio
            ax = axes[1, 0]
            grad_ratio = g_mean / (d_mean + 1e-8)
            ax.plot(epochs, grad_ratio, color=color, linewidth=2, label=display_name)
            lower = (g_mean - g_sem) / (d_mean + d_sem + 1e-8)
            upper = (g_mean + g_sem) / (d_mean - d_sem + 1e-8)
            ax.fill_between(epochs, lower, upper, color=color, alpha=0.2)
            
            # Plot Gradient Stability (Coefficient of Variation)
            ax = axes[1, 1]
            g_cv = g_std / (g_mean + 1e-8)
            d_cv = d_std / (d_mean + 1e-8)
            ax.plot(epochs, g_cv, color=color, linestyle='-', linewidth=1.5,
                   label=f'{display_name[:10]} G')
            ax.plot(epochs, d_cv, color=color, linestyle='--', linewidth=1.5,
                   label=f'{display_name[:10]} D')
        
        if not has_data:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No gradient data available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Format plots
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_title('Generator Gradient Norms', fontweight='bold')
        axes[0, 0].legend(loc='upper right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Discriminator Gradient Norms', fontweight='bold')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('G/D Gradient Ratio')
        axes[1, 0].set_title('Gradient Balance', fontweight='bold')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].set_title('Gradient Stability (CV)', fontweight='bold')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Gradient Dynamics - {dataset_name}', fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'gradient_norms_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Gradient norms saved to: {save_path}")
        return save_path
    
    def plot_failure_rate_analysis(self, results_df: pd.DataFrame,
                                  model_info: Dict,
                                  dataset_name: str,
                                  threshold: float = 0.3):
        """
        Figure 3: Failure rate analysis across multiple runs
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = list(model_info.keys())
        display_names = [model_info[m][0] for m in models if m in results_df['Model'].values]
        colors = [model_info[m][1] for m in models if m in results_df['Model'].values]
        
        # 1. Failure rate bar chart
        ax1 = axes[0]
        failure_rates = []
        
        for model in models:
            if model not in results_df['Model'].values:
                continue
            model_scores = results_df[results_df['Model'] == model]['Composite_Score'].values
            if len(model_scores) > 0:
                failures = np.sum(model_scores < threshold) / len(model_scores) * 100
                failure_rates.append(failures)
            else:
                failure_rates.append(0)
        
        if failure_rates:
            bars = ax1.bar(range(len(display_names)), failure_rates, color=colors, 
                          edgecolor='black', linewidth=1)
            ax1.set_xticks(range(len(display_names)))
            ax1.set_xticklabels(display_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, rate in zip(bars, failure_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        ax1.set_ylabel('Failure Rate (%)')
        ax1.set_title(f'Failure Rate (Score < {threshold})', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Score distribution with failure threshold
        ax2 = axes[1]
        for i, (model, color) in enumerate(zip(models, colors)):
            if model not in results_df['Model'].values:
                continue
            model_scores = results_df[results_df['Model'] == model]['Composite_Score'].values
            if len(model_scores) > 0:
                x_pos = [i] * len(model_scores)
                ax2.scatter(x_pos, model_scores, color=color, alpha=0.6, 
                          s=50, edgecolor='black', linewidth=0.5)
        
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold})')
        ax2.set_xticks(range(len(display_names)))
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Score Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Success probability (Kaplan-Meier style)
        ax3 = axes[2]
        for i, (model, color) in enumerate(zip(models, colors)):
            if model not in results_df['Model'].values:
                continue
            model_scores = results_df[results_df['Model'] == model]['Composite_Score'].values
            if len(model_scores) > 0:
                sorted_scores = np.sort(model_scores)
                success_prob = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
                ax3.step(sorted_scores, success_prob, where='post', 
                        color=color, linewidth=2, label=display_names[i])
        
        ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Composite Score')
        ax3.set_ylabel('Success Probability')
        ax3.set_title('Success Probability Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if display_names:
            ax3.legend(loc='lower right', fontsize=8)
        
        plt.suptitle(f'Failure Rate Analysis - {dataset_name}', fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'failure_rate_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Failure rate analysis saved to: {save_path}")
        return save_path
    
    def plot_convergence_speed(self, results_df: pd.DataFrame,
                              model_info: Dict,
                              dataset_name: str,
                              target_score: float = 0.7):
        """
        Figure 4: Convergence speed analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = list(model_info.keys())
        display_names = [model_info[m][0] for m in models if m in results_df['Model'].values]
        colors = [model_info[m][1] for m in models if m in results_df['Model'].values]
        
        # 1. Epochs to converge
        ax1 = axes[0]
        convergence_epochs = []
        convergence_stds = []
        
        for model in models:
            if model not in results_df['Model'].values:
                continue
            model_data = results_df[results_df['Model'] == model]
            epochs_to_target = []
            
            for run in model_data['Run'].unique():
                run_data = model_data[model_data['Run'] == run]
            if 'Epoch' in run_data.columns and 'Score' in run_data.columns:
                scores = run_data[run_data['Score'] >= target_score]
                if len(scores) > 0:
                    epochs_to_target.append(scores.iloc[0]['Epoch'])
            
            if epochs_to_target:
                convergence_epochs.append(np.mean(epochs_to_target))
                convergence_stds.append(np.std(epochs_to_target))
            else:
                convergence_epochs.append(999)
                convergence_stds.append(0)
        
        if convergence_epochs:
            bars = ax1.bar(range(len(display_names)), convergence_epochs, yerr=convergence_stds,
                          color=colors, capsize=5, edgecolor='black', linewidth=1)
            ax1.set_xticks(range(len(display_names)))
            ax1.set_xticklabels(display_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, epoch in zip(bars, convergence_epochs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{epoch:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        ax1.set_ylabel('Epochs to Converge')
        ax1.set_title(f'Convergence Speed (Target: {target_score})', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Learning curves comparison
        ax2 = axes[1]
        for i, (model, color) in enumerate(zip(models, colors)):
            if model not in results_df['Model'].values:
                continue
            model_data = results_df[results_df['Model'] == model]
            if 'Epoch' in model_data.columns and 'Score' in model_data.columns:
                # Average across runs
                pivot = model_data.pivot_table(index='Epoch', columns='Run', values='Score')
                mean_scores = pivot.mean(axis=1)
                std_scores = pivot.std(axis=1)
                
                epochs = mean_scores.index
                ax2.plot(epochs, mean_scores, color=color, linewidth=2,
                        label=display_names[i])
                ax2.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                                color=color, alpha=0.2)
        
        ax2.axhline(y=target_score, color='red', linestyle='--', linewidth=2,
                   label=f'Target ({target_score})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Learning Curves', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if display_names:
            ax2.legend(loc='lower right', fontsize=8)
        
        plt.suptitle(f'Convergence Analysis - {dataset_name}', fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'convergence_speed_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Convergence speed saved to: {save_path}")
        return save_path
    
    def plot_stability_heatmap(self, results_df: pd.DataFrame,
                              model_info: Dict,
                              dataset_name: str):
        """
        Figure 5: Stability heatmap across runs and metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        models = list(model_info.keys())
        display_names = [model_info[m][0] for m in models if m in results_df['Model'].values]
        
        metrics = ['Composite_Score', 'FID', 'Wasserstein', 'ACF_Similarity', 'PSD_Similarity']
        metric_display = ['Composite', 'FID', 'Wasserstein', 'ACF', 'PSD']
        
        # 1. Coefficient of Variation heatmap
        ax1 = axes[0]
        cv_matrix = []
        
        for model in models:
            if model not in results_df['Model'].values:
                continue
            model_cvs = []
            for metric in metrics:
                if metric in results_df.columns:
                    values = results_df[results_df['Model'] == model][metric].values
                    if len(values) > 0 and np.mean(values) != 0:
                        cv = np.std(values) / (np.mean(values) + 1e-8)
                    else:
                        cv = 0
                    model_cvs.append(cv)
                else:
                    model_cvs.append(0)
            cv_matrix.append(model_cvs)
        
        if cv_matrix:
            im1 = ax1.imshow(cv_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
            ax1.set_xticks(range(len(metric_display)))
            ax1.set_xticklabels(metric_display, rotation=45, ha='right')
            ax1.set_yticks(range(len(display_names)))
            ax1.set_yticklabels(display_names)
            ax1.set_title('Stability (Coefficient of Variation)', fontweight='bold')
            plt.colorbar(im1, ax=ax1, label='CV (lower is more stable)')
            
            # Add value annotations
            for i in range(len(display_names)):
                for j in range(len(metric_display)):
                    val = cv_matrix[i][j]
                    color = 'black' if val < 0.3 else 'white'
                    ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        else:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # 2. Success Rate heatmap (across thresholds)
        ax2 = axes[1]
        thresholds = np.linspace(0.3, 0.8, 6)
        success_matrix = []
        
        for model in models:
            if model not in results_df['Model'].values:
                continue
            model_success = []
            scores = results_df[results_df['Model'] == model]['Composite_Score'].values
            for thresh in thresholds:
                if len(scores) > 0:
                    success_rate = np.sum(scores >= thresh) / len(scores) * 100
                else:
                    success_rate = 0
                model_success.append(success_rate)
            success_matrix.append(model_success)
        
        if success_matrix:
            im2 = ax2.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            ax2.set_xticks(range(len(thresholds)))
            ax2.set_xticklabels([f'{t:.1f}' for t in thresholds])
            ax2.set_xlabel('Success Threshold')
            ax2.set_yticks(range(len(display_names)))
            ax2.set_yticklabels(display_names)
            ax2.set_title('Success Rate Across Thresholds (%)', fontweight='bold')
            plt.colorbar(im2, ax=ax2, label='Success Rate (%)')
            
            # Add value annotations
            for i in range(len(display_names)):
                for j in range(len(thresholds)):
                    val = success_matrix[i][j]
                    color = 'black' if val > 50 else 'white'
                    ax2.text(j, i, f'{val:.0f}', ha='center', va='center', color=color)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        plt.suptitle(f'Stability Analysis - {dataset_name}', fontweight='bold', y=1.05)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'stability_heatmap_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Stability heatmap saved to: {save_path}")
        return save_path
    
    def plot_comprehensive_dynamics(self, histories: Dict[str, List[Dict]],
                                   gradient_histories: Dict[str, List[Dict]],
                                   results_df: pd.DataFrame,
                                   model_info: Dict,
                                   dataset_name: str,
                                   model_names: List[str]):
        """
        Figure 6: Comprehensive 4-panel dynamics figure for paper
        """
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        display_names = [model_info[m][0] for m in model_names if m in histories]
        colors = [model_info[m][1] for m in model_names if m in histories]
        
        # Panel A: Loss curves
        ax1 = plt.subplot(gs[0, 0])
        
        for idx, (model_key, display_name, color) in enumerate(zip(model_names, display_names, colors)):
            if model_key not in histories:
                continue
            
            model_histories = histories[model_key]
            if not model_histories:
                continue
            
            # Extract G-loss across seeds
            g_losses = []
            for seed_history in model_histories[:5]:
                if seed_history and isinstance(seed_history, dict):
                    g_loss = seed_history.get('g_loss', [])
                    if len(g_loss) > 0:
                        g_losses.append(g_loss)
            
            if g_losses:
                min_len = min([len(g) for g in g_losses])
                g_losses = np.array([g[:min_len] for g in g_losses])
                g_mean = np.mean(g_losses, axis=0)
                g_sem = np.std(g_losses, axis=0) / np.sqrt(len(g_losses))
                epochs = range(min_len)
                
                # Smooth
                if len(g_mean) > 10:
                    from scipy.ndimage import gaussian_filter1d
                    g_mean = gaussian_filter1d(g_mean, sigma=1.0)
                
                ax1.plot(epochs, g_mean, color=color, linewidth=2, label=display_name)
                ax1.fill_between(epochs, g_mean - g_sem, g_mean + g_sem,
                                color=color, alpha=0.2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Generator Loss')
        ax1.set_title('A: Generator Loss Convergence', fontweight='bold', loc='left')
        if display_names:
            ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Gradient norms
        ax2 = plt.subplot(gs[0, 1])
        
        for idx, (model_key, display_name, color) in enumerate(zip(model_names, display_names, colors)):
            if model_key not in gradient_histories:
                continue
            
            grad_data = gradient_histories[model_key]
            if not grad_data:
                continue
            
            g_grads = []
            for seed_grads in grad_data[:5]:
                if seed_grads and isinstance(seed_grads, dict):
                    g_grad = seed_grads.get('g_grad_norm', [])
                    if len(g_grad) > 0:
                        g_grads.append(g_grad)
            
            if g_grads:
                min_len = min([len(g) for g in g_grads])
                g_grads = np.array([g[:min_len] for g in g_grads])
                g_mean = np.mean(g_grads, axis=0)
                g_sem = np.std(g_grads, axis=0) / np.sqrt(len(g_grads))
                epochs = range(min_len)
                
                ax2.plot(epochs, g_mean, color=color, linewidth=2, label=display_name)
                ax2.fill_between(epochs, g_mean - g_sem, g_mean + g_sem,
                                color=color, alpha=0.2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('B: Gradient Norm Evolution', fontweight='bold', loc='left')
        if display_names:
            ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Panel C: Failure rate
        ax3 = plt.subplot(gs[1, 0])
        
        thresholds = np.linspace(0.3, 0.8, 6)
        
        for idx, (model_key, display_name, color) in enumerate(zip(model_names, display_names, colors)):
            if model_key not in results_df['Model'].values:
                continue
            model_scores = results_df[results_df['Model'] == model_key]['Composite_Score'].values
            if len(model_scores) > 0:
                success_rates = [np.sum(model_scores >= t) / len(model_scores) * 100 for t in thresholds]
                ax3.plot(thresholds, success_rates, 'o-', color=color, linewidth=2, label=display_name)
        
        ax3.set_xlabel('Success Threshold')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('C: Success Rate Analysis', fontweight='bold', loc='left')
        if display_names:
            ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 105)
        
        # Panel D: Stability heatmap (simplified)
        ax4 = plt.subplot(gs[1, 1])
        
        stability_data = []
        for model_key in model_names:
            if model_key not in results_df['Model'].values:
                continue
            model_scores = results_df[results_df['Model'] == model_key]['Composite_Score'].values
            if len(model_scores) > 0:
                stability_data.append([
                    np.mean(model_scores),
                    np.std(model_scores),
                    np.min(model_scores),
                    np.max(model_scores),
                    np.percentile(model_scores, 25),
                    np.percentile(model_scores, 75)
                ])
            else:
                stability_data.append([0, 0, 0, 0, 0, 0])
        
        if stability_data:
            stability_data = np.array(stability_data).T
            im = ax4.imshow(stability_data, cmap='viridis', aspect='auto')
            ax4.set_yticks(range(6))
            ax4.set_yticklabels(['Mean', 'Std', 'Min', 'Max', '25%', '75%'])
            ax4.set_xticks(range(len(display_names)))
            ax4.set_xticklabels(display_names, rotation=45, ha='right')
            ax4.set_title('D: Stability Statistics', fontweight='bold', loc='left')
            plt.colorbar(im, ax=ax4, label='Score')
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        plt.suptitle(f'Comprehensive Training Dynamics - {dataset_name}', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'comprehensive_dynamics_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        print(f"  Comprehensive dynamics saved to: {save_path}")
        return save_path