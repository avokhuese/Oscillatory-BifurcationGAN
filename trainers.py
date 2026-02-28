# trainers.py
"""
Training scripts for all GAN variants
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from metrics import GANLosses, TimeSeriesMetrics, InceptionScore, FrechetInceptionDistance

class BaseGANTrainer:
    """Base trainer for all GAN variants"""
    
    def __init__(self, generator, discriminator, config, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            generator.parameters(), 
            lr=config.LEARNING_RATE, 
            betas=(config.BETA1, config.BETA2)
        )
        self.d_optimizer = optim.Adam(
            discriminator.parameters(), 
            lr=config.LEARNING_RATE, 
            betas=(config.BETA1, config.BETA2)
        )
        
        # Loss functions
        self.criterion = nn.BCELoss()
        self.gan_losses = GANLosses()
        
        # Metrics
        self.ts_metrics = TimeSeriesMetrics()
        self.inception_score = InceptionScore(device)
        self.fid = FrechetInceptionDistance(device)
        
        # History
        self.history = {
            'g_loss': [], 'd_loss': [], 'wasserstein_loss': [],
            'gradient_penalty': [], 'bifurcation_loss': []
        }
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        g_losses, d_losses = [], []
        
        for real_data, _ in dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real data
            real_validity = self.discriminator(real_data)
            real_labels = torch.ones(batch_size, 1).to(self.device)
            d_real_loss = self.criterion(real_validity, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            fake_validity = self.discriminator(fake_data.detach())
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            d_fake_loss = self.criterion(fake_validity, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            fake_validity = self.discriminator(fake_data)
            g_loss = self.criterion(fake_validity, real_labels)
            g_loss.backward()
            self.g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        
        return np.mean(g_losses), np.mean(d_losses)
    
    def train(self, train_loader, val_loader, n_epochs):
        """Full training loop"""
        for epoch in range(n_epochs):
            self.generator.train()
            self.discriminator.train()
            
            g_loss, d_loss = self.train_epoch(train_loader)
            
            # Validation
            self.generator.eval()
            self.discriminator.eval()
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            self.history['g_loss'].append(g_loss)
            self.history['d_loss'].append(d_loss)
            
            # Logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
                print(f"Val Metrics: MAE: {val_metrics.get('MAE', 0):.4f}")
    
    def evaluate(self, dataloader):
        """Evaluate on validation set"""
        all_real = []
        all_fake = []
        
        with torch.no_grad():
            for real_data, _ in dataloader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
                fake_data = self.generator(z)
                
                all_real.append(real_data.cpu().numpy())
                all_fake.append(fake_data.cpu().numpy())
        
        real_np = np.concatenate(all_real, axis=0)
        fake_np = np.concatenate(all_fake, axis=0)
        
        # Calculate metrics
        metrics = self.ts_metrics.calculate_basic_metrics(real_np, fake_np)
        metrics['Autocorr_Sim'] = self.ts_metrics.autocorrelation_similarity(real_np, fake_np)
        metrics['Wasserstein'] = self.ts_metrics.wasserstein_distance(real_np, fake_np)
        
        return metrics

class VanillaGANTrainer(BaseGANTrainer):
    """Vanilla GAN trainer"""
    pass

class WGANTrainer(BaseGANTrainer):
    """Wasserstein GAN trainer"""
    
    def train_epoch(self, dataloader):
        g_losses, d_losses = [], []
        
        for real_data, _ in dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator (critic)
            for _ in range(5):  # Train critic 5 times
                self.d_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
                fake_data = self.generator(z)
                
                real_validity = self.discriminator(real_data)
                fake_validity = self.discriminator(fake_data.detach())
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_loss.backward()
                self.d_optimizer.step()
                
                # Weight clipping
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            fake_validity = self.discriminator(fake_data)
            
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            self.g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        
        return np.mean(g_losses), np.mean(d_losses)

class WGANGPTrainer(BaseGANTrainer):
    """WGAN with Gradient Penalty"""
    
    def train_epoch(self, dataloader):
        g_losses, d_losses, gp_list = [], [], []
        
        for real_data, _ in dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            for _ in range(5):
                self.d_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
                fake_data = self.generator(z)
                
                real_validity = self.discriminator(real_data)
                fake_validity = self.discriminator(fake_data.detach())
                
                # Wasserstein loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                
                # Gradient penalty
                gp = self.gan_losses.gradient_penalty(
                    self.discriminator, real_data, fake_data, 
                    self.config.LAMBDA_GP
                )
                d_loss += gp
                
                d_loss.backward()
                self.d_optimizer.step()
                
                gp_list.append(gp.item())
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            fake_validity = self.discriminator(fake_data)
            
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            self.g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
        
        self.history['gradient_penalty'].append(np.mean(gp_list))
        return np.mean(g_losses), np.mean(d_losses)

class SigWANGPTrainer(WGANGPTrainer):
    """Sigmoid WGAN with Gradient Penalty"""
    pass

class SISVAETrainer(BaseGANTrainer):
    """SISVAE trainer"""
    pass

class VAETrainer(BaseGANTrainer):
    """VAE trainer"""
    pass

class GANVAETrainer(BaseGANTrainer):
    """GAN-VAE hybrid trainer"""
    pass

class OBGANTrainer(BaseGANTrainer):
    """Oscillatory Bifurcation GAN trainer"""
    
    def __init__(self, generator, discriminator, perturbation_fn, config, device):
        super().__init__(generator, discriminator, config, device)
        self.perturbation_fn = perturbation_fn.to(device)
        self.p_optimizer = optim.Adam(perturbation_fn.parameters(), lr=config.LEARNING_RATE)
        
    def train_epoch(self, dataloader):
        g_losses, d_losses, bif_losses, p_losses = [], [], [], []
        
        for real_data, _ in dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)
            
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            
            real_validity = self.discriminator(real_data)
            fake_validity = self.discriminator(fake_data.detach())
            
            # Add perturbation reward
            pert_reward = self.perturbation_fn(fake_data.detach().mean(dim=1))
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) - 0.1 * pert_reward.mean()
            
            # Gradient penalty
            gp = self.gan_losses.gradient_penalty(
                self.discriminator, real_data, fake_data, 
                self.config.LAMBDA_GP
            )
            d_loss += gp
            
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            fake_validity = self.discriminator(fake_data)
            
            # Generator loss with bifurcation regularization
            g_loss = -torch.mean(fake_validity)
            
            # Bifurcation loss
            bif_loss = self.gan_losses.bifurcation_loss(
                self.generator, real_data, self.config.LAMBDA_BIF
            )
            g_loss += bif_loss
            
            # Perturbation reward
            pert_reward = self.perturbation_fn(fake_data.mean(dim=1))
            g_loss -= 0.1 * pert_reward.mean()
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Train Perturbation function
            self.p_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.device)
            fake_data = self.generator(z)
            
            p_output = self.perturbation_fn(fake_data.mean(dim=1))
            p_loss = -torch.mean(p_output)  # Maximize reward for fake data
            
            p_loss.backward()
            self.p_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            bif_losses.append(bif_loss.item())
            p_losses.append(p_loss.item())
        
        self.history['bifurcation_loss'].append(np.mean(bif_losses))
        return np.mean(g_losses), np.mean(d_losses)