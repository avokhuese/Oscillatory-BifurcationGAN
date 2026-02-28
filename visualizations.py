
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
import pandas as pd
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
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

class PublicationVisualizer:
    """
    Generate publication-quality visualizations for the paper
    """
    
    def __init__(self, save_dir: str = './paper_figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_time_series_comparison(self, real_data: np.ndarray, 
                                    synthetic_data: Dict[str, np.ndarray],
                                    dataset_name: str,
                                    n_samples: int = 3):
        """
        Figure 1: Time series comparison between real and synthetic data
        """
        n_models = len(synthetic_data)
        fig, axes = plt.subplots(n_samples, n_models + 1, 
                                 figsize=(4 * (n_models + 1), 3 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_models + 1))
        
        for i in range(n_samples):
            # Real data
            ax = axes[i, 0]
            ax.plot(real_data[i, :, 0], color='black', linewidth=1.5, alpha=0.8)
            ax.set_ylabel(f'Sample {i+1}', fontsize=10)
            if i == 0:
                ax.set_title('Real Data', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Synthetic data from each model
            for j, (model_name, synth_data) in enumerate(synthetic_data.items()):
                ax = axes[i, j + 1]
                ax.plot(synth_data[i, :, 0], color=colors[j + 1], 
                       linewidth=1.5, alpha=0.8, linestyle='--')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.set_title(model_name.replace('_', ' ').title(), fontweight='bold')
        
        plt.suptitle(f'Time Series Comparison: {dataset_name}', fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig1_timeseries_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_distribution_comparison(self, real_data: np.ndarray,
                                     synthetic_data: Dict[str, np.ndarray],
                                     dataset_name: str):
        """
        Figure 2: Distribution comparison (histograms and KDE)
        """
        n_models = len(synthetic_data)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))
        
        # Flatten data for distribution
        real_flat = real_data[:, :, 0].flatten()
        
        # Histograms
        ax_hist = axes[0, 0]
        ax_hist.hist(real_flat, bins=50, density=True, alpha=0.7, 
                    color='black', edgecolor='black', linewidth=0.5)
        ax_hist.set_title('Real Data', fontweight='bold')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Density')
        ax_hist.grid(True, alpha=0.3)
        
        # KDE plots
        ax_kde = axes[1, 0]
        from scipy.stats import gaussian_kde
        kde_real = gaussian_kde(real_flat)
        x_range = np.linspace(real_flat.min(), real_flat.max(), 200)
        ax_kde.plot(x_range, kde_real(x_range), color='black', linewidth=2, label='Real')
        ax_kde.set_xlabel('Value')
        ax_kde.set_ylabel('Density')
        ax_kde.grid(True, alpha=0.3)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        
        for j, (model_name, synth_data) in enumerate(synthetic_data.items()):
            synth_flat = synth_data[:, :, 0].flatten()
            
            # Histogram
            ax_hist = axes[0, j + 1]
            ax_hist.hist(synth_flat, bins=50, density=True, alpha=0.7,
                        color=colors[j], edgecolor='black', linewidth=0.5)
            ax_hist.set_title(model_name.replace('_', ' ').title(), fontweight='bold')
            ax_hist.set_xlabel('Value')
            ax_hist.grid(True, alpha=0.3)
            
            # KDE
            ax_kde = axes[1, j + 1]
            kde_synth = gaussian_kde(synth_flat)
            ax_kde.plot(x_range, kde_synth(x_range), color=colors[j], 
                       linewidth=2, label=model_name)
            ax_kde.plot(x_range, kde_real(x_range), color='black', 
                       linewidth=1.5, linestyle='--', alpha=0.7, label='Real')
            ax_kde.set_xlabel('Value')
            ax_kde.grid(True, alpha=0.3)
            ax_kde.legend(fontsize=8)
        
        plt.suptitle(f'Distribution Analysis: {dataset_name}', fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig2_distribution_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_temporal_dynamics(self, real_data: np.ndarray,
                               synthetic_data: Dict[str, np.ndarray],
                               dataset_name: str):
        """
        Figure 3: Temporal dynamics analysis (ACF, PSD, Phase Space)
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        n_models = len(synthetic_data)
        
        # 1. Autocorrelation Function
        ax_acf = plt.subplot(gs[0, :])
        max_lag = min(50, real_data.shape[1] // 2)
        
        # Compute ACF for real data
        real_acf = self._compute_acf(real_data[:, :, 0], max_lag)
        lags = np.arange(max_lag)
        ax_acf.plot(lags, real_acf, color='black', linewidth=2.5, label='Real', zorder=10)
        
        # Compute ACF for synthetic data
        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        for j, (model_name, synth_data) in enumerate(synthetic_data.items()):
            synth_acf = self._compute_acf(synth_data[:, :, 0], max_lag)
            ax_acf.plot(lags, synth_acf, color=colors[j], linewidth=1.5, 
                       linestyle='--', label=model_name.replace('_', ' ').title())
        
        ax_acf.set_xlabel('Lag', fontsize=12)
        ax_acf.set_ylabel('Autocorrelation', fontsize=12)
        ax_acf.set_title('Autocorrelation Function', fontweight='bold', fontsize=14)
        ax_acf.grid(True, alpha=0.3)
        ax_acf.legend(loc='upper right', fontsize=10, ncol=2)
        
        # 2. Power Spectral Density
        ax_psd = plt.subplot(gs[1, :])
        
        # Compute PSD for real data
        freqs, real_psd = signal.welch(real_data[:, :, 0].flatten(), fs=1.0, nperseg=256)
        ax_psd.loglog(freqs[1:], real_psd[1:], color='black', linewidth=2.5, label='Real', zorder=10)
        
        # Compute PSD for synthetic data
        for j, (model_name, synth_data) in enumerate(synthetic_data.items()):
            _, synth_psd = signal.welch(synth_data[:, :, 0].flatten(), fs=1.0, nperseg=256)
            ax_psd.loglog(freqs[1:], synth_psd[1:], color=colors[j], 
                         linewidth=1.5, linestyle='--', 
                         label=model_name.replace('_', ' ').title())
        
        ax_psd.set_xlabel('Frequency', fontsize=12)
        ax_psd.set_ylabel('Power Spectral Density', fontsize=12)
        ax_psd.set_title('Power Spectrum', fontweight='bold', fontsize=14)
        ax_psd.grid(True, alpha=0.3, which='both')
        ax_psd.legend(loc='lower left', fontsize=10)
        
        # 3. Phase Space Reconstruction (for first sample)
        from mpl_toolkits.mplot3d import Axes3D
        ax_phase = plt.subplot(gs[2, 0], projection='3d')
        self._plot_phase_space(ax_phase, real_data[0, :, 0], 'Real', 'black')
        
        # 4-6. Phase space for top models
        ax_phase2 = plt.subplot(gs[2, 1], projection='3d')
        ax_phase3 = plt.subplot(gs[2, 2], projection='3d')
        
        top_models = list(synthetic_data.keys())[:2]  # First two models
        self._plot_phase_space(ax_phase2, synthetic_data[top_models[0]][0, :, 0], 
                              top_models[0].replace('_', ' ').title(), colors[0])
        self._plot_phase_space(ax_phase3, synthetic_data[top_models[1]][0, :, 0], 
                              top_models[1].replace('_', ' ').title(), colors[1])
        
        plt.suptitle(f'Temporal Dynamics Analysis: {dataset_name}', fontweight='bold', y=1.02, fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig3_temporal_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def _compute_acf(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute average autocorrelation across samples"""
        n_samples = data.shape[0]
        acf_sum = np.zeros(max_lag)
        
        for i in range(min(n_samples, 50)):  # Limit to 50 samples
            series = data[i]
            series = series - np.mean(series)
            acf = np.correlate(series, series, mode='full')[len(series)-1:len(series)+max_lag-1]
            if acf[0] != 0:
                acf = acf / acf[0]  # Normalize
            acf_sum += acf[:max_lag]
        
        return acf_sum / min(n_samples, 50)
    
    def _plot_phase_space(self, ax, series: np.ndarray, title: str, color: str):
        """Plot 3D phase space reconstruction"""
        # Time-delay embedding
        tau = 3
        dim = 3
        if len(series) > dim * tau:
            embedded = np.column_stack([
                series[:-2*tau],
                series[tau:-tau],
                series[2*tau:]
            ])
            ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                   color=color, linewidth=1, alpha=0.8)
        
        ax.set_xlabel('x(t)')
        ax.set_ylabel('x(t+τ)')
        ax.set_zlabel('x(t+2τ)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_model_comparison(self, benchmark_results: pd.DataFrame, dataset_name: str):
        """
        Figure 4: Model comparison bar chart with statistical significance
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        models = benchmark_results['Model'].values
        scores = benchmark_results['Composite_Score'].values
        stds = benchmark_results.get('Composite_Score_std', np.zeros_like(scores))
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        models = [models[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        stds = [stds[i] for i in sorted_idx]
        
        # Bar plot
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))
        bars = axes[0].barh(range(len(models)), scores, xerr=stds, 
                           color=colors, capsize=5, edgecolor='black', linewidth=1)
        
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels([m.replace('_', ' ').title() for m in models])
        axes[0].set_xlabel('Composite Score', fontsize=12)
        axes[0].set_title(f'Model Performance: {dataset_name}', fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
            axes[0].text(score + std + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{score:.3f}±{std:.3f}', va='center', fontsize=9)
        
        # Box plot for score distribution
        score_data = []
        for model in models:
            model_scores = benchmark_results[benchmark_results['Model'] == model]['Composite_Score'].values
            score_data.append(model_scores)
        
        bp = axes[1].boxplot(score_data, patch_artist=True, labels=[m.replace('_', ' ').title() for m in models])
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('Composite Score', fontsize=12)
        axes[1].set_title('Score Distribution Across Runs', fontweight='bold', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig4_comparison_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_radar_comparison(self, benchmark_results: pd.DataFrame, dataset_name: str):
        """
        Figure 5: Radar chart comparing models across multiple metrics
        """
        # Select metrics for radar chart
        metrics = ['FID', 'Wasserstein', 'ACF_Similarity', 'PSD_Similarity', 'JS_Divergence']
        n_metrics = len(metrics)
        
        # Create angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Get top 5 models
        top_models = benchmark_results.nlargest(5, 'Composite_Score')['Model'].values
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))
        
        for i, model in enumerate(top_models):
            values = []
            model_data = benchmark_results[benchmark_results['Model'] == model].iloc[0]
            
            for metric in metrics:
                if metric in model_data.index:
                    val = model_data[metric]
                    # Normalize for radar chart
                    if metric in ['FID', 'Wasserstein', 'JS_Divergence']:
                        val = 1 / (1 + val)  # Lower is better
                    values.append(val)
                else:
                    values.append(0)
            
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], 
                   label=model.replace('_', ' ').title())
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(f'Multi-Metric Comparison: {dataset_name}', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig5_radar_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_bifurcation_dynamics(self, generator: nn.Module, device: torch.device,
                                   dataset_name: str):
        """
        Figure 6: Visualization of bifurcation dynamics
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Generate samples with different bifurcation parameters
        bifurcation_params = [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
        
        with torch.no_grad():
            for idx, mu in enumerate(bifurcation_params):
                i, j = idx // 3, idx % 3
                ax = axes[i, j]
                
                # Modify bifurcation parameter
                original_mu = None
                if hasattr(generator, 'bifurcation_layers'):
                    for layer in generator.bifurcation_layers:
                        if hasattr(layer, 'mu'):
                            original_mu = layer.mu.data.clone()
                            layer.mu.data.fill_(mu)
                
                # Generate sample
                z = torch.randn(1, generator.latent_dim, device=device)
                sample = generator(z).cpu().numpy()[0, :, 0]
                
                # Plot
                ax.plot(sample, color='black', linewidth=1.5)
                ax.set_title(f'μ = {mu:.1f}', fontweight='bold')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1.5, 1.5)
                
                # Restore parameter
                if original_mu is not None:
                    for layer in generator.bifurcation_layers:
                        if hasattr(layer, 'mu'):
                            layer.mu.data = original_mu
        
        plt.suptitle(f'Bifurcation Dynamics: Effect of μ Parameter', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig6_bifurcation_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_oscillator_dynamics(self, generator: nn.Module, device: torch.device,
                                  dataset_name: str):
        """
        Figure 7: Visualization of coupled oscillator dynamics
        """
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        with torch.no_grad():
            # Generate sample
            z = torch.randn(1, generator.latent_dim, device=device)
            sample = generator(z).cpu().numpy()[0, :, 0]
            
            # Time series
            ax1 = plt.subplot(gs[0, 0])
            ax1.plot(sample, color='black', linewidth=1.5)
            ax1.set_title('Generated Time Series', fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            
            # Spectrum
            ax2 = plt.subplot(gs[0, 1])
            freqs, psd = signal.welch(sample, fs=1.0, nperseg=128)
            ax2.semilogy(freqs, psd, color='blue', linewidth=1.5)
            ax2.set_title('Power Spectrum', fontweight='bold')
            ax2.set_xlabel('Frequency')
            ax2.set_ylabel('Power')
            ax2.grid(True, alpha=0.3)
            
            # Phase space
            from mpl_toolkits.mplot3d import Axes3D
            ax3 = plt.subplot(gs[1, 0], projection='3d')
            tau = 3
            if len(sample) > 3 * tau:
                embedded = np.column_stack([
                    sample[:-2*tau],
                    sample[tau:-tau],
                    sample[2*tau:]
                ])
                ax3.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                         color='red', linewidth=1)
            ax3.set_title('Phase Space', fontweight='bold')
            ax3.set_xlabel('x(t)')
            ax3.set_ylabel('x(t+τ)')
            ax3.set_zlabel('x(t+2τ)')
            ax3.grid(True, alpha=0.3)
            
            # Autocorrelation
            ax4 = plt.subplot(gs[1, 1])
            acf = self._compute_acf(sample.reshape(1, -1), 50)
            ax4.plot(acf, color='green', linewidth=1.5)
            ax4.set_title('Autocorrelation', fontweight='bold')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Correlation')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(f'Oscillator Dynamics Analysis', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig7_oscillator_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def plot_tsne_visualization(self, real_data: np.ndarray,
                                synthetic_data: Dict[str, np.ndarray],
                                dataset_name: str):
        """
        Figure 8: t-SNE visualization of real vs synthetic data manifolds
        """
        fig, axes = plt.subplots(1, len(synthetic_data) + 1, figsize=(5 * (len(synthetic_data) + 1), 5))
        
        # Prepare data
        n_samples_per_model = min(200, real_data.shape[0], 
                                 min([s.shape[0] for s in synthetic_data.values()]))
        real_sample = real_data[:n_samples_per_model].reshape(n_samples_per_model, -1)
        
        # Combine all data for t-SNE
        all_data = [real_sample]
        labels = [0] * n_samples_per_model
        colors = ['black']
        model_names = ['Real']
        
        for j, (model_name, synth_data) in enumerate(synthetic_data.items()):
            synth_sample = synth_data[:n_samples_per_model].reshape(n_samples_per_model, -1)
            all_data.append(synth_sample)
            labels.extend([j + 1] * n_samples_per_model)
            colors.append(plt.cm.Set1(j))
            model_names.append(model_name.replace('_', ' ').title())
        
        all_data = np.vstack(all_data)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        data_tsne = tsne.fit_transform(all_data)
        
        # Plot
        for i, (start, end) in enumerate([(i * n_samples_per_model, (i + 1) * n_samples_per_model) 
                                          for i in range(len(model_names))]):
            ax = axes[i]
            ax.scatter(data_tsne[start:end, 0], data_tsne[start:end, 1], 
                      c=colors[i], s=10, alpha=0.7, label=model_names[i])
            ax.set_title(model_names[i], fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f't-SNE Manifold Visualization: {dataset_name}', fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f'fig8_tsne_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def create_paper_figure_grid(self, dataset_name: str):
        """
        Create a comprehensive figure grid for the paper
        """
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # This would be populated with the actual plots
        # Placeholder for now
        for i in range(4):
            for j in range(3):
                ax = plt.subplot(gs[i, j])
                ax.text(0.5, 0.5, f'Panel {chr(65 + i*3 + j)}', 
                       ha='center', va='center', fontsize=20)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle(f'Comprehensive Analysis: {dataset_name}', fontweight='bold', fontsize=18, y=0.98)
        
        save_path = os.path.join(self.save_dir, f'fig_grid_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path
    
    def generate_all_figures(self, real_data: np.ndarray,
                             synthetic_data: Dict[str, np.ndarray],
                             benchmark_results: Optional[pd.DataFrame] = None,
                             generator: Optional[nn.Module] = None,
                             device: Optional[torch.device] = None,
                             dataset_name: str = 'default'):
        """
        Generate all paper figures
        """
        print(f"\nGenerating all figures for {dataset_name}...")
        
        figures = {}
        
        # Figure 1: Time series comparison
        figures['fig1'] = self.plot_time_series_comparison(
            real_data, synthetic_data, dataset_name
        )
        
        # Figure 2: Distribution comparison
        figures['fig2'] = self.plot_distribution_comparison(
            real_data, synthetic_data, dataset_name
        )
        
        # Figure 3: Temporal dynamics
        figures['fig3'] = self.plot_temporal_dynamics(
            real_data, synthetic_data, dataset_name
        )
        
        # Figure 4: Model comparison
        if benchmark_results is not None:
            figures['fig4'] = self.plot_model_comparison(
                benchmark_results, dataset_name
            )
        
        # Figure 5: Radar comparison
        if benchmark_results is not None:
            figures['fig5'] = self.plot_radar_comparison(
                benchmark_results, dataset_name
            )
        
        # Figure 6: Bifurcation dynamics
        if generator is not None and device is not None:
            figures['fig6'] = self.plot_bifurcation_dynamics(
                generator, device, dataset_name
            )
        
        # Figure 7: Oscillator dynamics
        if generator is not None and device is not None:
            figures['fig7'] = self.plot_oscillator_dynamics(
                generator, device, dataset_name
            )
        
        # Figure 8: t-SNE visualization
        figures['fig8'] = self.plot_tsne_visualization(
            real_data, synthetic_data, dataset_name
        )
        
        print(f"\nAll figures saved to: {self.save_dir}")
        
        return figures