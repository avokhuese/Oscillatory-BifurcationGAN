"""
Ablation Study Module for Oscillatory BifurcationGAN
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

from config import config
from models import (
    OscillatoryBifurcationGenerator, 
    OscillatoryBifurcationDiscriminator,
    VanillaGenerator, VanillaDiscriminator,
    WGANGenerator, WGANDiscriminator
)
from gan_framework import GANTrainer, create_trainer
from evaluation import Evaluator

class AblationStudy:
    """
    Comprehensive ablation study for Oscillatory BifurcationGAN
    Analyzes contribution of each component to overall performance
    """
    
    def __init__(self, dataset_name: str, train_loader, val_loader, test_loader, 
                 n_features: int, seq_len: int, device: torch.device, save_dir: str):
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = device
        self.save_dir = os.path.join(save_dir, 'ablation_studies', dataset_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Get real test data for evaluation
        self.real_test = self._get_test_data()
        self.evaluator = Evaluator(self.real_test)
        
        # Store results
        self.results = {}
        self.training_histories = {}
        
    def _get_test_data(self) -> np.ndarray:
        """Extract test data from loader"""
        test_data = []
        for batch in self.test_loader:
            if isinstance(batch, (list, tuple)):
                test_data.append(batch[0].numpy())
            else:
                test_data.append(batch.numpy())
        return np.concatenate(test_data, axis=0)
    
    def run_component_ablation(self, epochs: int = 100):
        """
        Run ablation study by removing/enabling different components
        """
        print("\n" + "="*80)
        print(f"ABLATION STUDY: {self.dataset_name.upper()}")
        print("="*80)
        
        # Define ablation configurations
        ablation_configs = [
            {
                'name': 'Full Oscillatory BifurcationGAN',
                'use_bifurcation': True,
                'use_oscillators': True,
                'use_coupling': True,
                'use_multiscale': True,
                'use_noise': True,
                'description': 'Complete model with all components'
            },
            {
                'name': 'No Bifurcation',
                'use_bifurcation': False,
                'use_oscillators': True,
                'use_coupling': True,
                'use_multiscale': True,
                'use_noise': True,
                'description': 'Remove bifurcation dynamics'
            },
            {
                'name': 'No Oscillators',
                'use_bifurcation': True,
                'use_oscillators': False,
                'use_coupling': False,
                'use_multiscale': True,
                'use_noise': True,
                'description': 'Remove coupled oscillators'
            },
            {
                'name': 'No Coupling',
                'use_bifurcation': True,
                'use_oscillators': True,
                'use_coupling': False,
                'use_multiscale': True,
                'use_noise': True,
                'description': 'Remove oscillator coupling'
            },
            {
                'name': 'No Multiscale',
                'use_bifurcation': True,
                'use_oscillators': True,
                'use_coupling': True,
                'use_multiscale': False,
                'use_noise': True,
                'description': 'Remove multiscale processing'
            },
            {
                'name': 'No Noise',
                'use_bifurcation': True,
                'use_oscillators': True,
                'use_coupling': True,
                'use_multiscale': True,
                'use_noise': False,
                'description': 'Remove phase noise'
            },
            {
                'name': 'Vanilla GAN Baseline',
                'use_bifurcation': False,
                'use_oscillators': False,
                'use_coupling': False,
                'use_multiscale': False,
                'use_noise': False,
                'description': 'Basic GAN without any enhancements'
            }
        ]
        
        # Run each configuration
        for cfg in ablation_configs:
            print(f"\n{'-'*60}")
            print(f"Testing: {cfg['name']}")
            print(f"Description: {cfg['description']}")
            print(f"{'-'*60}")
            
            # Run multiple trials for statistical significance
            trial_results = []
            trial_histories = []
            
            for trial in range(3):  # 3 trials per configuration
                print(f"  Trial {trial+1}/3...")
                
                # Create model with this configuration
                generator, discriminator = self._create_ablation_model(cfg)
                
                # Train model
                trainer = GANTrainer(
                    model_type="oscillatory_bifurcation_gan",
                    generator=generator,
                    discriminator=discriminator,
                    n_features=self.n_features,
                    seq_len=self.seq_len,
                    device=self.device
                )
                
                # Modify training parameters based on config
                history = trainer.train(
                    self.train_loader, 
                    self.val_loader,
                    epochs=min(epochs, 50)  # Reduced for ablation
                )
                
                # Generate samples
                with torch.no_grad():
                    z = torch.randn(500, config.LATENT_DIM, device=self.device)
                    samples = generator(z, self.seq_len).cpu().numpy()
                
                # Evaluate
                metrics = self.evaluator.calculate_all_metrics(
                    samples, f"{cfg['name']}_trial{trial}"
                )
                trial_results.append(metrics)
                trial_histories.append(history)
                
                # Clean up
                del generator, discriminator, trainer
                torch.cuda.empty_cache()
            
            # Aggregate results
            avg_metrics = self._aggregate_results(trial_results)
            self.results[cfg['name']] = {
                'metrics': avg_metrics,
                'config': cfg,
                'trials': trial_results,
                'histories': trial_histories
            }
            
            print(f"  Composite Score: {avg_metrics['Composite_Score']:.4f} ± "
                  f"{np.std([r['Composite_Score'] for r in trial_results]):.4f}")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _create_ablation_model(self, config_dict: Dict) -> Tuple[nn.Module, nn.Module]:
        """Create model with specific ablation configuration"""
        
        if config_dict['name'] == 'Vanilla GAN Baseline':
            generator = VanillaGenerator(config.LATENT_DIM, self.n_features, self.seq_len)
            discriminator = VanillaDiscriminator(self.n_features, self.seq_len)
        else:
            # Create modified oscillatory bifurcation model
            generator = self._create_modified_generator(config_dict)
            discriminator = OscillatoryBifurcationDiscriminator(self.n_features, self.seq_len)
        
        return generator.to(self.device), discriminator.to(self.device)
    
    def _create_modified_generator(self, config_dict: Dict) -> nn.Module:
        """Create generator with specific components enabled/disabled"""
        
        class ModifiedOscillatoryGenerator(nn.Module):
            def __init__(self, cfg, n_features, seq_len):
                super().__init__()
                self.cfg = cfg
                self.n_features = n_features
                self.seq_len = seq_len
                self.hidden_dim = config.GENERATOR_HIDDEN
                
                # Base generator components
                self.noise_processor = nn.Sequential(
                    nn.Linear(config.LATENT_DIM, self.hidden_dim * 4),
                    nn.LayerNorm(self.hidden_dim * 4),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
                    nn.LayerNorm(self.hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.LeakyReLU(0.2)
                )
                
                # Conditional components
                if cfg['use_multiscale']:
                    self.temporal_convs = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                            nn.InstanceNorm1d(self.hidden_dim),
                            nn.LeakyReLU(0.2)
                        ) for _ in range(2)
                    ])
                
                if cfg['use_oscillators']:
                    self.n_oscillators = min(config.N_OSCILLATORS, self.hidden_dim // 8)
                    self.to_oscillator = nn.Linear(self.hidden_dim, self.n_oscillators * 2)
                    self.from_oscillator = nn.Linear(self.n_oscillators * 2, self.hidden_dim)
                    
                    if cfg['use_coupling']:
                        self.coupling = nn.Parameter(
                            torch.randn(self.n_oscillators, self.n_oscillators) * config.OSCILLATOR_COUPLING
                        )
                    
                    self.frequencies = nn.Parameter(
                        torch.tensor(config.NATURAL_FREQUENCIES[:self.n_oscillators])
                    )
                
                if cfg['use_bifurcation']:
                    self.mu = nn.Parameter(torch.tensor(config.HOPF_MU))
                    self.alpha = nn.Parameter(torch.tensor(config.HOPF_ALPHA))
                
                # Output projection
                self.output_projection = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.LayerNorm(self.hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim // 2, n_features),
                    nn.Tanh()
                )
                
                # Positional encoding
                self.register_buffer('pos_encoding', self._create_pos_encoding(seq_len))
            
            def _create_pos_encoding(self, seq_len):
                position = torch.arange(seq_len).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                                   -(np.log(10000.0) / self.hidden_dim))
                pe = torch.zeros(seq_len, self.hidden_dim)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe.unsqueeze(0)
            
            def forward(self, z, seq_len=None):
                batch_size = z.size(0)
                seq_len = seq_len or self.seq_len
                
                # Process noise
                h = self.noise_processor(z)
                h = h.unsqueeze(1).expand(-1, seq_len, -1)
                h = h + self.pos_encoding[:, :seq_len, :]
                
                # Multiscale processing
                if hasattr(self, 'temporal_convs'):
                    h_t = h.transpose(1, 2)
                    for conv in self.temporal_convs:
                        h_t = h_t + conv(h_t) * 0.1
                    h = h_t.transpose(1, 2)
                
                # Oscillator dynamics
                if hasattr(self, 'to_oscillator'):
                    osc_state = self.to_oscillator(h)
                    phase = osc_state[:, :, :self.n_oscillators]
                    amplitude = osc_state[:, :, self.n_oscillators:]
                    
                    # Simple oscillator update
                    t = torch.arange(seq_len, device=z.device).float().view(1, seq_len, 1)
                    
                    if hasattr(self, 'coupling'):
                        # With coupling
                        phase_update = self.frequencies.view(1, 1, -1) * t
                        if self.cfg['use_noise'] and self.training:
                            phase_update = phase_update + torch.randn_like(phase_update) * 0.1
                    else:
                        # Without coupling
                        phase_update = self.frequencies.view(1, 1, -1) * t
                    
                    phase = phase + torch.sin(phase_update) * 0.1
                    
                    # Bifurcation in amplitude
                    if hasattr(self, 'mu'):
                        amplitude = amplitude + self.mu * amplitude - amplitude**3
                    
                    # Combine
                    combined = torch.cat([phase, amplitude], dim=-1)
                    h = h + self.from_oscillator(combined) * 0.1
                
                # Output
                output = self.output_projection(h)
                
                return output
        
        return ModifiedOscillatoryGenerator(config_dict, self.n_features, self.seq_len)
    
    def _aggregate_results(self, trial_results: List[Dict]) -> Dict:
        """Aggregate results across trials with statistics"""
        aggregated = {}
        
        # Get all metric keys
        keys = trial_results[0].keys()
        
        for key in keys:
            values = [r[key] for r in trial_results if not np.isnan(r[key])]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def _save_results(self):
        """Save ablation study results"""
        # Save metrics
        results_path = os.path.join(self.save_dir, 'ablation_results.json')
        
        # Convert to serializable format
        serializable_results = {}
        for name, data in self.results.items():
            serializable_results[name] = {
                'metrics': {k: float(v) for k, v in data['metrics'].items()},
                'config': data['config'],
                'composite_score': float(data['metrics']['Composite_Score']),
                'composite_std': float(data['metrics'].get('Composite_Score_std', 0))
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nAblation results saved to: {results_path}")
    
    def plot_ablation_results(self):
        """Create comprehensive ablation study plots - FIXED VERSION"""
        
        # Extract data
        names = list(self.results.keys())
        scores = [self.results[n]['metrics']['Composite_Score'] for n in names]
        stds = [self.results[n]['metrics'].get('Composite_Score_std', 0) for n in names]
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        names = [names[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        
        # 1. Bar plot with error bars
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main bar plot
        ax1 = plt.subplot(gs[0, 0])
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        bars = ax1.barh(range(len(names)), scores, xerr=stds, color=colors, 
                    capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=10)
        ax1.set_xlabel('Composite Score', fontsize=12)
        ax1.set_title(f'Ablation Study: {self.dataset_name}', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
            ax1.text(score + std + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}±{std:.3f}', va='center', fontsize=9)
        
        # 2. Component contribution analysis
        ax2 = plt.subplot(gs[0, 1])
        
        # Calculate component contributions
        full_name = 'Full Oscillatory BifurcationGAN'
        if full_name in names:
            full_score = scores[names.index(full_name)]
        else:
            # Find the configuration with most components enabled
            full_score = max(scores)
        
        components = ['Bifurcation', 'Oscillators', 'Coupling', 'Multiscale', 'Noise']
        contributions = []
        
        for component in components:
            component_name = f'No {component}'
            if component_name in names:
                component_score = scores[names.index(component_name)]
                contribution = (full_score - component_score) / full_score * 100
                contributions.append(max(0, contribution))  # Ensure non-negative
            else:
                contributions.append(0)
        
        colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(components)))
        bars2 = ax2.bar(components, contributions, color=colors2, alpha=0.8,
                        edgecolor='black', linewidth=1)
        
        ax2.set_ylabel('Performance Drop (%)', fontsize=12)
        ax2.set_title('Component Importance Analysis', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, contrib in zip(bars2, contributions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{contrib:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Statistical significance heatmap
        ax3 = plt.subplot(gs[1, 0])
        
        # Create p-value matrix
        n_models = len(names)
        p_values = np.zeros((n_models, n_models))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i == j:
                    p_values[i, j] = 1.0
                else:
                    # Compute t-test between configurations
                    if name1 in self.results and name2 in self.results:
                        trials1 = [t['Composite_Score'] for t in self.results[name1]['trials']]
                        trials2 = [t['Composite_Score'] for t in self.results[name2]['trials']]
                        
                        if len(trials1) > 1 and len(trials2) > 1:
                            from scipy import stats
                            _, p_val = stats.ttest_ind(trials1, trials2)
                            p_values[i, j] = p_val
                        else:
                            p_values[i, j] = 0.5  # Default if insufficient data
        
        im = ax3.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
        ax3.set_xticks(range(n_models))
        ax3.set_yticks(range(n_models))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_title('Statistical Significance (p-values)', fontweight='bold', fontsize=14)
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, label='p-value')
        
        # Add significance markers
        for i in range(n_models):
            for j in range(n_models):
                if p_values[i, j] < 0.05 and i != j:
                    ax3.text(j, i, '*', ha='center', va='center', color='black', fontsize=12)
        
        # 4. Radar chart of top configurations - FIXED
        ax4 = plt.subplot(gs[1, 1], projection='polar')
        
        # Select top 5 configurations
        top_n = min(5, len(names))
        top_names = names[:top_n]
        
        # Metrics for radar chart
        radar_metrics = ['FID', 'Wasserstein', 'ACF_Similarity', 'PSD_Similarity', 'JS_Divergence']
        n_metrics = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        colors_radar = plt.cm.tab10(np.linspace(0, 1, top_n))
        
        for i, name in enumerate(top_names):
            values = []
            for metric in radar_metrics:
                if metric in self.results[name]['metrics']:
                    val = self.results[name]['metrics'][metric]
                    # Normalize for radar chart
                    if metric in ['FID', 'Wasserstein', 'JS_Divergence']:
                        val = 1 / (1 + val)  # Lower is better
                    values.append(val)
                else:
                    values.append(0)
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=name[:20], color=colors_radar[i], alpha=0.8)
            ax4.fill(angles, values, alpha=0.1, color=colors_radar[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(radar_metrics, fontsize=10)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top Configurations - Radar Chart', fontweight='bold', fontsize=14, pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax4.grid(True)
        
        plt.suptitle(f'Ablation Study: {self.dataset_name.upper()}', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, 'ablation_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Ablation plot saved to: {save_path}")
    
    def generate_component_analysis_table(self) -> pd.DataFrame:
        """Generate LaTeX-ready table of component contributions"""
        
        rows = []
        
        for name, data in self.results.items():
            config = data['config']
            metrics = data['metrics']
            
            row = {
                'Configuration': name,
                'Bifurcation': '✓' if config['use_bifurcation'] else '✗',
                'Oscillators': '✓' if config['use_oscillators'] else '✗',
                'Coupling': '✓' if config['use_coupling'] else '✗',
                'Multiscale': '✓' if config['use_multiscale'] else '✗',
                'Noise': '✓' if config['use_noise'] else '✗',
                'Composite Score': f"{metrics['Composite_Score']:.3f}",
                '±': f"±{metrics.get('Composite_Score_std', 0):.3f}",
                'FID': f"{metrics.get('FID', 0):.3f}",
                'Wasserstein': f"{metrics.get('Wasserstein', 0):.3f}",
                'ACF Sim': f"{metrics.get('ACF_Similarity', 0):.3f}",
                'PSD Sim': f"{metrics.get('PSD_Similarity', 0):.3f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by composite score
        df = df.sort_values('Composite Score', ascending=False)
        
        # Save CSV
        csv_path = os.path.join(self.save_dir, 'ablation_table.csv')
        df.to_csv(csv_path, index=False)
        
        # Save LaTeX
        latex_path = os.path.join(self.save_dir, 'ablation_table.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"\nAblation table saved to: {latex_path}")
        
        return df