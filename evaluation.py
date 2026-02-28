import numpy as np
import torch
from scipy import stats, signal
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from config import config

class Evaluator:
    """Evaluation framework for univariate time series"""
    
    def __init__(self, real_data: np.ndarray):
        """
        Args:
            real_data: shape (n_samples, n_timesteps, n_features)
        """
        self.real_data = real_data
        self.n_samples, self.n_timesteps, self.n_features = real_data.shape
        
        print(f"Evaluator initialized: {self.n_samples} samples, "
              f"{self.n_timesteps} timesteps, {self.n_features} features")
    
    def _align_shapes(self, real: np.ndarray, fake: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align shapes of real and fake data"""
        n_real, n_fake = real.shape[0], fake.shape[0]
        n_samples = min(n_real, n_fake)
        
        real = real[:n_samples]
        fake = fake[:n_samples]
        
        # Ensure same sequence length
        seq_len = min(real.shape[1], fake.shape[1])
        real = real[:, :seq_len, :]
        fake = fake[:, :seq_len, :]
        
        return real, fake
    
    def calculate_reconstruction_metrics(self, fake_data: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction metrics"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        real_flat = real.reshape(-1, self.n_features)
        fake_flat = fake.reshape(-1, self.n_features)
        
        metrics = {}
        
        # MAE
        metrics['MAE'] = mean_absolute_error(real_flat, fake_flat)
        
        # MSE
        metrics['MSE'] = mean_squared_error(real_flat, fake_flat)
        
        # RMSE
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        
        # MAPE
        mape = np.mean(np.abs((real_flat - fake_flat) / (np.abs(real_flat) + 1e-10))) * 100
        metrics['MAPE'] = mape
        
        return metrics
    
    def calculate_distribution_metrics(self, fake_data: np.ndarray) -> Dict[str, float]:
        """Calculate distribution similarity metrics"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        metrics = {}
        
        # Jensen-Shannon Divergence
        js_divergences = []
        for f in range(self.n_features):
            real_f = real[:, :, f].flatten()
            fake_f = fake[:, :, f].flatten()
            
            # Create histograms
            bins = np.linspace(min(real_f.min(), fake_f.min()), 
                              max(real_f.max(), fake_f.max()), 50)
            real_hist, _ = np.histogram(real_f, bins=bins, density=True)
            fake_hist, _ = np.histogram(fake_f, bins=bins, density=True)
            
            # Add epsilon to avoid zeros
            real_hist = real_hist + 1e-10
            fake_hist = fake_hist + 1e-10
            real_hist = real_hist / real_hist.sum()
            fake_hist = fake_hist / fake_hist.sum()
            
            # Calculate JS divergence
            M = (real_hist + fake_hist) / 2
            js_div = 0.5 * (stats.entropy(real_hist, M) + stats.entropy(fake_hist, M))
            js_divergences.append(js_div)
        
        metrics['JS_Divergence'] = np.mean(js_divergences)
        
        # Kolmogorov-Smirnov test
        ks_stats = []
        for f in range(self.n_features):
            real_f = real[:, :, f].flatten()
            fake_f = fake[:, :, f].flatten()
            ks_stat, _ = stats.ks_2samp(real_f, fake_f)
            ks_stats.append(ks_stat)
        
        metrics['KS_Statistic'] = np.mean(ks_stats)
        
        # Wasserstein distance
        wasserstein_dists = []
        for f in range(self.n_features):
            real_f = real[:, :, f].flatten()
            fake_f = fake[:, :, f].flatten()
            
            # Sort values
            real_sorted = np.sort(real_f)
            fake_sorted = np.sort(fake_f)
            
            # Calculate Wasserstein distance (simplified)
            wasserstein = np.mean(np.abs(real_sorted - fake_sorted))
            wasserstein_dists.append(wasserstein)
        
        metrics['Wasserstein_Distance'] = np.mean(wasserstein_dists)
        
        return metrics
    
    def calculate_temporal_metrics(self, fake_data: np.ndarray) -> Dict[str, float]:
        """Calculate temporal dynamics metrics"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        metrics = {}
        
        # Autocorrelation similarity
        max_lag = min(20, self.n_timesteps // 2)
        acf_similarities = []
        
        for f in range(self.n_features):
            real_acfs, fake_acfs = [], []
            
            # Calculate ACF for multiple samples
            n_samples_acf = min(10, real.shape[0])
            for i in range(n_samples_acf):
                real_series = real[i, :, f]
                fake_series = fake[i, :, f]
                
                # Calculate ACF
                real_acf = self._calculate_acf(real_series, max_lag)
                fake_acf = self._calculate_acf(fake_series, max_lag)
                
                real_acfs.append(real_acf)
                fake_acfs.append(fake_acf)
            
            # Calculate similarity
            real_acf_mean = np.mean(real_acfs, axis=0)
            fake_acf_mean = np.mean(fake_acfs, axis=0)
            
            # Normalize
            real_acf_norm = real_acf_mean / (np.max(np.abs(real_acf_mean)) + 1e-10)
            fake_acf_norm = fake_acf_mean / (np.max(np.abs(fake_acf_mean)) + 1e-10)
            
            similarity = 1 - np.mean(np.abs(real_acf_norm - fake_acf_norm))
            acf_similarities.append(similarity)
        
        metrics['ACF_Similarity'] = np.mean(acf_similarities)
        
        # Power spectral density similarity
        psd_similarities = []
        
        for f in range(self.n_features):
            real_psds, fake_psds = [], []
            
            n_samples_psd = min(5, real.shape[0])
            for i in range(n_samples_psd):
                real_series = real[i, :, f]
                fake_series = fake[i, :, f]
                
                # Calculate PSD using Welch's method
                freqs_real, psd_real = signal.welch(real_series, nperseg=min(256, len(real_series)))
                freqs_fake, psd_fake = signal.welch(fake_series, nperseg=min(256, len(fake_series)))
                
                # Interpolate to common frequency grid
                freqs_common = np.linspace(0, max(freqs_real[-1], freqs_fake[-1]), 100)
                psd_real_interp = np.interp(freqs_common, freqs_real, psd_real)
                psd_fake_interp = np.interp(freqs_common, freqs_fake, psd_fake)
                
                real_psds.append(psd_real_interp)
                fake_psds.append(psd_fake_interp)
            
            # Calculate similarity
            real_psd_mean = np.mean(real_psds, axis=0)
            fake_psd_mean = np.mean(fake_psds, axis=0)
            
            # Normalize
            real_psd_norm = real_psd_mean / (real_psd_mean.sum() + 1e-10)
            fake_psd_norm = fake_psd_mean / (fake_psd_mean.sum() + 1e-10)
            
            similarity = 1 - np.mean(np.abs(real_psd_norm - fake_psd_norm))
            psd_similarities.append(similarity)
        
        metrics['PSD_Similarity'] = np.mean(psd_similarities)
        
        return metrics
    
    def _calculate_acf(self, series: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function"""
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)
        
        if var == 0:
            return np.zeros(max_lag)
        
        acf = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag < n:
                acf[lag-1] = np.mean((series[:n-lag] - mean) * (series[lag:] - mean)) / var
        
        return acf
    
    def calculate_diversity_metrics(self, fake_data: np.ndarray) -> Dict[str, float]:
        """Calculate diversity metrics"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        metrics = {}
        
        # Intra-sample diversity
        fake_diversities = []
        n_samples_div = min(20, fake.shape[0])
        
        for f in range(self.n_features):
            fake_f = fake[:, :, f]
            distances = []
            for i in range(n_samples_div):
                for j in range(i + 1, n_samples_div):
                    dist = np.linalg.norm(fake_f[i] - fake_f[j])
                    distances.append(dist)
            
            if distances:
                fake_diversities.append(np.mean(distances))
        
        metrics['Intra_Sample_Diversity'] = np.mean(fake_diversities) if fake_diversities else 0
        
        return metrics
    
    def calculate_composite_score(self, fake_data: np.ndarray) -> float:
        """Calculate composite evaluation score"""
        recon_metrics = self.calculate_reconstruction_metrics(fake_data)
        dist_metrics = self.calculate_distribution_metrics(fake_data)
        temp_metrics = self.calculate_temporal_metrics(fake_data)
        div_metrics = self.calculate_diversity_metrics(fake_data)
        
        # Normalize scores (lower is better metrics)
        recon_score = 1 / (1 + recon_metrics['MAE'] + recon_metrics['MSE'])
        dist_score = 1 - min(dist_metrics['JS_Divergence'], 1)
        temp_score = (temp_metrics['ACF_Similarity'] + temp_metrics['PSD_Similarity']) / 2
        div_score = min(div_metrics['Intra_Sample_Diversity'], 1)
        
        # Weighted combination
        composite = (0.3 * recon_score + 0.3 * dist_score + 
                    0.2 * temp_score + 0.2 * div_score)
        
        return composite
    
    def calculate_all_metrics(self, fake_data: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """Calculate all evaluation metrics"""
        print(f"\nEvaluating {model_name}...")
        
        metrics = {}
        
        if config.CALCULATE_FID:
            metrics['FID'] = self._calculate_fid_simple(fake_data)
            print(f"  FID: {metrics['FID']:.4f}")
        
        if config.CALCULATE_MMD:
            metrics['MMD'] = self._calculate_mmd_simple(fake_data)
            print(f"  MMD: {metrics['MMD']:.4f}")
        
        if config.CALCULATE_WASSERSTEIN:
            metrics['Wasserstein'] = self.calculate_distribution_metrics(fake_data)['Wasserstein_Distance']
            print(f"  Wasserstein: {metrics['Wasserstein']:.4f}")
        
        if config.CALCULATE_JSD:
            metrics['JS_Divergence'] = self.calculate_distribution_metrics(fake_data)['JS_Divergence']
            print(f"  JS Divergence: {metrics['JS_Divergence']:.4f}")
        
        if config.CALCULATE_KS_TEST:
            metrics['KS_Statistic'] = self.calculate_distribution_metrics(fake_data)['KS_Statistic']
            print(f"  KS Statistic: {metrics['KS_Statistic']:.4f}")
        
        if config.CALCULATE_ACF_SIMILARITY:
            metrics['ACF_Similarity'] = self.calculate_temporal_metrics(fake_data)['ACF_Similarity']
            print(f"  ACF Similarity: {metrics['ACF_Similarity']:.3f}")
        
        if config.CALCULATE_PSD_SIMILARITY:
            metrics['PSD_Similarity'] = self.calculate_temporal_metrics(fake_data)['PSD_Similarity']
            print(f"  PSD Similarity: {metrics['PSD_Similarity']:.3f}")
        
        # Composite score
        composite = self.calculate_composite_score(fake_data)
        metrics['Composite_Score'] = composite
        print(f"  Composite Score: {composite:.3f}")
        
        return metrics
    
    def _calculate_fid_simple(self, fake_data: np.ndarray) -> float:
        """Simple FID-like metric"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        # Extract features (simple statistics)
        def extract_features(data):
            features = []
            for f in range(data.shape[-1]):
                series = data[:, :, f]
                features.extend([
                    np.mean(series, axis=1),
                    np.std(series, axis=1),
                    np.median(series, axis=1),
                    np.max(series, axis=1) - np.min(series, axis=1)
                ])
            return np.column_stack(features)
        
        real_feat = extract_features(real)
        fake_feat = extract_features(fake)
        
        # Calculate means and covariances
        mu_real = np.mean(real_feat, axis=0)
        mu_fake = np.mean(fake_feat, axis=0)
        sigma_real = np.cov(real_feat, rowvar=False)
        sigma_fake = np.cov(fake_feat, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_fake
        covmean = self._sqrtm(sigma_real @ sigma_fake)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return max(fid, 0)
    
    def _calculate_mmd_simple(self, fake_data: np.ndarray) -> float:
        """Simple MMD metric"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        # Flatten
        real_flat = real.reshape(real.shape[0], -1)
        fake_flat = fake.reshape(fake.shape[0], -1)
        
        # Subsample
        n_subsample = min(100, real_flat.shape[0], fake_flat.shape[0])
        real_sub = real_flat[np.random.choice(real_flat.shape[0], n_subsample, replace=False)]
        fake_sub = fake_flat[np.random.choice(fake_flat.shape[0], n_subsample, replace=False)]
        
        # RBF kernel MMD
        gamma = 1.0 / real_flat.shape[1]
        
        def rbf_kernel(x, y):
            dist = torch.cdist(torch.tensor(x), torch.tensor(y)).numpy()
            return np.exp(-gamma * dist ** 2)
        
        k_real_real = rbf_kernel(real_sub, real_sub)
        k_fake_fake = rbf_kernel(fake_sub, fake_sub)
        k_real_fake = rbf_kernel(real_sub, fake_sub)
        
        mmd = (np.mean(k_real_real) + np.mean(k_fake_fake) - 2 * np.mean(k_real_fake))
        
        return max(mmd, 0)
    
    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Matrix square root"""
        u, s, v = np.linalg.svd(matrix)
        return u @ np.diag(np.sqrt(s)) @ v
    
    def plot_comparison(self, fake_data: np.ndarray, model_name: str = "Model", save_path: str = None):
        """Plot comparison between real and generated data"""
        real, fake = self._align_shapes(self.real_data, fake_data)
        
        # Select a few samples
        n_samples_plot = min(3, real.shape[0])
        n_features_plot = min(3, self.n_features)
        
        fig, axes = plt.subplots(n_samples_plot, n_features_plot, 
                                figsize=(5 * n_features_plot, 4 * n_samples_plot))
        
        if n_samples_plot == 1:
            axes = axes.reshape(1, -1)
        if n_features_plot == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples_plot):
            for j in range(n_features_plot):
                ax = axes[i, j]
                
                # Plot real and fake
                ax.plot(real[i, :, j], label='Real', alpha=0.8, linewidth=2)
                ax.plot(fake[i, :, j], label='Generated', alpha=0.7, linestyle='--')
                
                ax.set_title(f'Sample {i+1}, Feature {j+1}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Real vs {model_name} Generated Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

class BenchmarkEvaluator:
    """Benchmark evaluator for comparing multiple models"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, dataset_name: str, model_name: str, metrics: Dict[str, float]):
        """Add evaluation result for a model on a dataset"""
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        self.results[dataset_name][model_name] = metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results"""
        summary = {}
        
        for dataset_name, models in self.results.items():
            summary[dataset_name] = {}
            for model_name, metrics in models.items():
                summary[dataset_name][model_name] = metrics.get('Composite_Score', 0)
        
        return summary
    
    def save_results(self, save_dir: str):
        """Save results to file"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save summary
        summary = self.get_summary()
        import json
        with open(os.path.join(save_dir, 'benchmark_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create comparison table
        self._create_comparison_table(save_dir)
    
    def _create_comparison_table(self, save_dir: str):
        """Create comparison table"""
        import pandas as pd
        
        rows = []
        for dataset_name, models in self.results.items():
            for model_name, metrics in models.items():
                row = {'Dataset': dataset_name, 'Model': model_name}
                row.update(metrics)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
        
        # Pivot table
        pivot = df.pivot_table(
            index='Model', 
            columns='Dataset', 
            values='Composite_Score',
            aggfunc='mean'
        )
        pivot['Average'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('Average', ascending=False)
        
        pivot.to_csv(os.path.join(save_dir, 'summary_table.csv'))
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(pivot.round(4).to_string())
        print("="*60)
        
        return pivot