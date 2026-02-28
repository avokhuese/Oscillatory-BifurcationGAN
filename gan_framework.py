from datetime import datetime
import json
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os

import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import config
from models import create_model, count_parameters

# ===================== LOSS FUNCTIONS =====================

class GANLoss(nn.Module):
    """Loss functions for different GAN variants"""
    
    def __init__(self, model_type: str):
        super().__init__()
        self.model_type = model_type
        
    def generator_loss(self, fake_validity: torch.Tensor) -> torch.Tensor:
        if self.model_type in ['wgan_gp', 'sig_wgan']:
            return -torch.mean(fake_validity)
        else:
            return nn.BCEWithLogitsLoss()(fake_validity, torch.ones_like(fake_validity))
    
    def discriminator_loss(self, real_validity: torch.Tensor, 
                          fake_validity: torch.Tensor) -> torch.Tensor:
        if self.model_type in ['wgan_gp', 'sig_wgan']:
            return -torch.mean(real_validity) + torch.mean(fake_validity)
        else:
            real_loss = nn.BCEWithLogitsLoss()(real_validity, torch.ones_like(real_validity))
            fake_loss = nn.BCEWithLogitsLoss()(fake_validity, torch.zeros_like(fake_validity))
            return (real_loss + fake_loss) / 2

class VAELoss(nn.Module):
    """VAE loss functions"""
    
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.kl_weight = config.KL_WEIGHT
        
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, 
                mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = self.reconstruction_loss(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        return recon_loss + self.kl_weight * kl_loss

# ===================== GRADIENT PENALTY =====================

def compute_gradient_penalty(discriminator, real_samples: torch.Tensor, 
                            fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated = interpolated.detach().requires_grad_(True)
    
    # Discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    grad_outputs = torch.ones_like(d_interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

# ===================== TRAINER CLASSES =====================

class GANTrainer:
    """Trainer for GAN models with history saving"""
    
    def __init__(self, model_type: str, generator: nn.Module, discriminator: nn.Module, 
                 n_features: int, seq_len: int, device: torch.device):
        self.model_type = model_type
        self.generator = generator
        self.discriminator = discriminator
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = device
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            generator.parameters(),
            lr=config.GENERATOR_LR,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.optimizer_D = optim.Adam(
            discriminator.parameters(),
            lr=config.DISCRIMINATOR_LR,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Loss functions
        self.gan_loss = GANLoss(model_type)
        
        # Training history - add gradient norm tracking
        self.history = {
            'g_loss': [], 
            'd_loss': [], 
            'wasserstein': [],
            'g_grad_norm': [],  # Add this
            'd_grad_norm': []   # Add this
        }
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == 'cuda' else None
    
    def train_step(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """Single training step with gradient norm tracking"""
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        
        # ================== Train Discriminator ==================
        self.optimizer_D.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Generate fake samples
            z = torch.randn(batch_size, config.LATENT_DIM, device=self.device)
            fake_batch = self.generator(z, self.seq_len)
            
            # Ensure shapes match
            if fake_batch.shape != real_batch.shape:
                fake_batch = fake_batch[:, :real_batch.size(1), :real_batch.size(2)]
            
            # Real and fake predictions
            real_validity = self.discriminator(real_batch)
            fake_validity = self.discriminator(fake_batch.detach())
            
            # Discriminator loss
            d_loss = self.gan_loss.discriminator_loss(real_validity, fake_validity)
            
            # Add gradient penalty for WGAN-GP
            if self.model_type in ['wgan_gp', 'sig_wgan'] and config.USE_GRADIENT_PENALTY:
                gp = compute_gradient_penalty(self.discriminator, real_batch, fake_batch, self.device)
                d_loss = d_loss + config.LAMBDA_GP * gp
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.optimizer_D)
        else:
            d_loss.backward()
            self.optimizer_D.step()
        
        # ================== Train Generator ==================
        self.optimizer_G.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Generate new fake samples
            z = torch.randn(batch_size, config.LATENT_DIM, device=self.device)
            fake_batch = self.generator(z, self.seq_len)
            fake_validity = self.discriminator(fake_batch)
            
            # Generator loss
            g_loss = self.gan_loss.generator_loss(fake_validity)
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            g_loss.backward()
            self.optimizer_G.step()
        
        # Calculate metrics
        with torch.no_grad():
            real_validity = self.discriminator(real_batch)
            fake_validity = self.discriminator(fake_batch)
            wasserstein = torch.mean(real_validity) - torch.mean(fake_validity)
            
            # Calculate gradient norms
            g_grad_norm = 0
            d_grad_norm = 0
            for p in self.generator.parameters():
                if p.grad is not None:
                    g_grad_norm += p.grad.norm().item() ** 2
            for p in self.discriminator.parameters():
                if p.grad is not None:
                    d_grad_norm += p.grad.norm().item() ** 2
            
            g_grad_norm = np.sqrt(g_grad_norm)
            d_grad_norm = np.sqrt(d_grad_norm)
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'wasserstein': wasserstein.item(),
            'g_grad_norm': g_grad_norm,
            'd_grad_norm': d_grad_norm
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = None, 
              save_history: bool = False, save_dir: str = None):  # ADD THESE PARAMETERS
        """Main training loop with automatic history saving"""
        epochs = epochs or config.EPOCHS
        
        # Create save directory if saving history
        if save_history and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_dir = os.path.join(save_dir, f"{self.model_type}_{timestamp}")
            os.makedirs(model_save_dir, exist_ok=True)
        else:
            model_save_dir = None
        
        print(f"\nTraining {self.model_type}...")
        
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            
            epoch_metrics = {'g_loss': [], 'd_loss': [], 'wasserstein': [],
                           'g_grad_norm': [], 'd_grad_norm': []}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    real_batch = batch[0]
                else:
                    real_batch = batch
                
                if real_batch.size(0) < 2:
                    continue
                
                try:
                    metrics = self.train_step(real_batch)
                    for k, v in metrics.items():
                        if k in epoch_metrics:
                            epoch_metrics[k].append(v)
                    
                    pbar.set_postfix({
                        'g': np.mean(epoch_metrics['g_loss'][-10:]),
                        'd': np.mean(epoch_metrics['d_loss'][-10:])
                    })
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            # Store epoch averages
            for k in epoch_metrics:
                if epoch_metrics[k]:
                    self.history[k].append(np.mean(epoch_metrics[k]))
                else:
                    self.history[k].append(0.0)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: G={self.history['g_loss'][-1]:.4f}, "
                      f"D={self.history['d_loss'][-1]:.4f}")
            
            # Save checkpoint periodically
            if save_history and model_save_dir and (epoch + 1) % 10 == 0:
                self.save_history(os.path.join(model_save_dir, f"epoch_{epoch+1}"))
        
        # Save final history
        if save_history and model_save_dir:
            self.save_history(os.path.join(model_save_dir, "final"))
            print(f"\nTraining history saved to: {model_save_dir}")
        
        return self.history
    
    def save_history(self, save_path: str):
        """Save training history to file"""
        history_data = {
            'g_loss': self.history.get('g_loss', []),
            'd_loss': self.history.get('d_loss', []),
            'wasserstein': self.history.get('wasserstein', []),
            'g_grad_norm': self.history.get('g_grad_norm', []),
            'd_grad_norm': self.history.get('d_grad_norm', [])
        }
        
        with open(f"{save_path}_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"  History saved to: {save_path}_history.json")
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, config.LATENT_DIM, device=self.device)
            samples = self.generator(z, self.seq_len)
        return samples.cpu()
    
class VAETrainer:
    """Trainer for VAE models"""
    
    def __init__(self, vae_model: nn.Module, n_features: int, seq_len: int, device: torch.device):
        self.vae = vae_model
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = device
        
        self.optimizer = optim.Adam(
            vae_model.parameters(),
            lr=config.GENERATOR_LR,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.vae_loss = VAELoss()
        self.history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step for VAE"""
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = self.vae(batch)
        
        # Compute loss
        recon_loss = nn.MSELoss()(recon, batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + config.KL_WEIGHT * kl_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = None,
              save_history: bool = False, save_dir: str = None):  # ADD THESE
        """Main training loop for VAE"""
        epochs = epochs or config.EPOCHS
        
        if save_history and save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nTraining VAE...")
        
        for epoch in range(epochs):
            # ... rest of training code ...
            
            if save_history and save_dir and (epoch + 1) % 10 == 0:
                self.save_history(os.path.join(save_dir, f"epoch_{epoch+1}"))
        
        if save_history and save_dir:
            self.save_history(os.path.join(save_dir, "final"))
        
        return self.history
    
    def save_history(self, save_path: str):
        """Save VAE training history"""
        history_data = {
            'loss': self.history.get('loss', []),
            'recon_loss': self.history.get('recon_loss', []),
            'kl_loss': self.history.get('kl_loss', [])
        }
        
        with open(f"{save_path}_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"  History saved to: {save_path}_history.json")
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        self.vae.eval()
        with torch.no_grad():
            samples = self.vae.generate(n_samples, self.device)
        return samples.cpu()

class VAEGANTrainer:
    """Trainer for VAE-GAN models"""
    
    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 n_features: int, seq_len: int, device: torch.device):
        self.generator = generator
        self.discriminator = discriminator
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = device
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            generator.parameters(),
            lr=config.GENERATOR_LR,
            betas=(config.BETA1, config.BETA2)
        )
        
        self.optimizer_D = optim.Adam(
            discriminator.parameters(),
            lr=config.DISCRIMINATOR_LR,
            betas=(config.BETA1, config.BETA2)
        )
        
        # Loss functions
        self.vae_loss = VAELoss()
        self.gan_loss = GANLoss('vanilla_gan')
        
        # Training history
        self.history = {'g_loss': [], 'd_loss': [], 'vae_loss': [], 'total_g_loss': []}
    
    def train_step(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """Single training step for VAE-GAN"""
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        
        # ================== Train Discriminator ==================
        self.optimizer_D.zero_grad()
        
        # Generate fake samples
        recon, mu, logvar, z = self.generator(real_batch)
        
        # Real and fake predictions
        real_validity = self.discriminator(real_batch)
        fake_validity = self.discriminator(recon.detach())
        
        # Discriminator loss
        d_loss = self.gan_loss.discriminator_loss(real_validity, fake_validity)
        d_loss.backward()
        self.optimizer_D.step()
        
        # ================== Train Generator ==================
        self.optimizer_G.zero_grad()
        
        # Reconstruction
        recon, mu, logvar, z = self.generator(real_batch)
        
        # VAE loss
        vae_loss = self.vae_loss(recon, real_batch, mu, logvar)
        
        # Adversarial loss
        fake_validity = self.discriminator(recon)
        g_adv_loss = self.gan_loss.generator_loss(fake_validity)
        
        # Total generator loss
        total_g_loss = vae_loss + g_adv_loss * 0.1
        total_g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'd_loss': d_loss.item(),
            'vae_loss': vae_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'total_g_loss': total_g_loss.item()
        }
    
    def train(self, train_loader, val_loader=None, epochs: int = None):
        """Main training loop"""
        epochs = epochs or config.EPOCHS
        
        print(f"\nTraining VAE-GAN...")
        
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            
            epoch_metrics = {'d_loss': [], 'vae_loss': [], 'total_g_loss': []}
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                if batch.size(0) < 2:
                    continue
                
                try:
                    metrics = self.train_step(batch)
                    epoch_metrics['d_loss'].append(metrics['d_loss'])
                    epoch_metrics['vae_loss'].append(metrics['vae_loss'])
                    epoch_metrics['total_g_loss'].append(metrics['total_g_loss'])
                    
                    pbar.set_postfix({
                        'D': np.mean(epoch_metrics['d_loss'][-10:]),
                        'VAE': np.mean(epoch_metrics['vae_loss'][-10:])
                    })
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            # Store epoch averages
            for k in epoch_metrics:
                if epoch_metrics[k]:
                    self.history[k].append(np.mean(epoch_metrics[k]))
                else:
                    self.history[k].append(0.0)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: D={self.history['d_loss'][-1]:.4f}, "
                      f"VAE={self.history['vae_loss'][-1]:.4f}")
        
        return self.history
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, config.VAE_LATENT_DIM, device=self.device)
            samples = self.generator.decode(z)
        return samples.cpu()
    def save_history(self, save_path: str):
        """Save training history to file"""
        history_data = {
            'g_loss': self.history.get('g_loss', []),
            'd_loss': self.history.get('d_loss', []),
            'wasserstein': self.history.get('wasserstein', [])
        }
        
        grad_data = {
            'g_grad_norm': self.history.get('g_grad_norm', []),
            'd_grad_norm': self.history.get('d_grad_norm', [])
        }
        
        with open(f"{save_path}_history.json", 'w') as f:
            json.dump(history_data, f)
        
        with open(f"{save_path}_gradients.json", 'w') as f:
            json.dump(grad_data, f)
# ===================== MAIN FACTORY =====================

def create_trainer(model_type: str, n_features: int, seq_len: int, device: torch.device):
    """Factory function to create appropriate trainer"""
    
    generator, discriminator = create_model(model_type, n_features, seq_len, device)
    
    if model_type == "sisvae":
        trainer = VAETrainer(generator, n_features, seq_len, device)
    elif model_type == "vae_gan":
        trainer = VAEGANTrainer(generator, discriminator, n_features, seq_len, device)
    else:
        trainer = GANTrainer(model_type, generator, discriminator, n_features, seq_len, device)
    
    return trainer