# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Tuple, List, Optional, Dict, Any
# from config import config

# # ===================== BASE MODELS =====================

# class BaseGenerator(nn.Module):
#     """Base generator class"""
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.n_features = n_features
#         self.seq_len = seq_len

# class BaseDiscriminator(nn.Module):
#     """Base discriminator class"""
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__()
#         self.n_features = n_features
#         self.seq_len = seq_len

# # ===================== OSCILLATORY BIFURCATION GAN =====================

# class OscillatoryBifurcationLayer(nn.Module):
#     """Oscillatory bifurcation dynamics layer"""
    
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.hidden_dim = hidden_dim
        
#         # Bifurcation parameters
#         self.mu = nn.Parameter(torch.tensor(config.HOPF_MU))
#         self.omega = nn.Parameter(torch.tensor(config.HOPF_OMEGA))
#         self.alpha = nn.Parameter(torch.tensor(config.HOPF_ALPHA))
        
#         # Oscillator parameters
#         self.frequencies = nn.Parameter(torch.tensor(config.NATURAL_FREQUENCIES[:config.N_OSCILLATORS]))
#         self.coupling = nn.Parameter(torch.randn(config.N_OSCILLATORS, config.N_OSCILLATORS) * config.OSCILLATOR_COUPLING)
        
#         # Transformations
#         self.to_oscillator = nn.Linear(hidden_dim, config.N_OSCILLATORS * 2)
#         self.from_oscillator = nn.Linear(config.N_OSCILLATORS * 2, hidden_dim)
        
#     def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
        
#         # Project to oscillator space
#         osc_state = self.to_oscillator(x)
#         phase = osc_state[:, :, :self.frequencies.size(0)]
#         amplitude = osc_state[:, :, self.frequencies.size(0):]
        
#         # Apply oscillatory bifurcation dynamics
#         for i in range(1, seq_len):
#             # Phase dynamics (Kuramoto with bifurcation)
#             phase_diff = phase[:, i-1:i, :] - phase[:, i-1:i, :].transpose(1, 2)
#             coupling_term = torch.matmul(torch.sin(phase_diff), self.coupling)
            
#             dphase = self.frequencies + coupling_term.squeeze(1)
#             if config.PHASE_NOISE_STD > 0:
#                 dphase = dphase + torch.randn_like(dphase) * config.PHASE_NOISE_STD
            
#             # Amplitude dynamics with Hopf bifurcation
#             damp = -config.AMPLITUDE_DECAY * amplitude[:, i-1:i, :]
#             damp = damp + self.mu * amplitude[:, i-1:i, :] - amplitude[:, i-1:i, :]**3
            
#             # Update
#             phase[:, i:i+1, :] = phase[:, i-1:i, :] + dphase.unsqueeze(1) * 0.1
#             amplitude[:, i:i+1, :] = amplitude[:, i-1:i, :] + damp * 0.1
            
#             # Saturate
#             amplitude[:, i:i+1, :] = torch.tanh(amplitude[:, i:i+1, :]) * config.AMPLITUDE_SATURATION
        
#         # Combine and project back
#         combined = torch.cat([phase, amplitude], dim=-1)
#         output = self.from_oscillator(combined)
        
#         return x + output * 0.1

# class OscillatoryBifurcationGenerator(BaseGenerator):
#     """Oscillatory BifurcationGAN Generator"""
    
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__(latent_dim, n_features, seq_len)
#         self.hidden_dim = config.GENERATOR_HIDDEN
        
#         # Noise processing
#         self.noise_processor = nn.Sequential(
#             nn.Linear(latent_dim, self.hidden_dim * 4),
#             nn.LayerNorm(self.hidden_dim * 4),
#             nn.LeakyReLU(0.2),
#             nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
#             nn.LayerNorm(self.hidden_dim * 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.LayerNorm(self.hidden_dim),
#             nn.LeakyReLU(0.2)
#         )
        
#         # Temporal convolutions
#         self.temporal_convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
#                 nn.InstanceNorm1d(self.hidden_dim),
#                 nn.LeakyReLU(0.2)
#             ) for _ in range(3)
#         ])
        
#         # Oscillatory bifurcation layers
#         self.bifurcation_layers = nn.ModuleList([
#             OscillatoryBifurcationLayer(self.hidden_dim)
#             for _ in range(2)
#         ])
        
#         # Output projection
#         self.output_projection = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim // 2),
#             nn.LeakyReLU(0.2),
#             nn.Linear(self.hidden_dim // 2, n_features),
#             nn.Tanh()
#         )
        
#         # Positional encoding
#         self.register_buffer('pos_encoding', self._create_pos_encoding(seq_len))
        
#     def _create_pos_encoding(self, seq_len):
#         position = torch.arange(seq_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
#                            -(np.log(10000.0) / self.hidden_dim))
#         pe = torch.zeros(seq_len, self.hidden_dim)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)
    
#     def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         batch_size = z.size(0)
#         seq_len = seq_len or self.seq_len
        
#         # Process noise
#         h = self.noise_processor(z)
#         h = h.unsqueeze(1).repeat(1, seq_len, 1)
        
#         # Add positional encoding
#         h = h + self.pos_encoding[:, :seq_len, :].to(h.device)
        
#         # Apply temporal convolutions
#         h = h.transpose(1, 2)
#         for conv in self.temporal_convs:
#             h = h + conv(h)
#         h = h.transpose(1, 2)
        
#         # Apply oscillatory bifurcation dynamics
#         t = torch.arange(seq_len, device=z.device).float().view(1, seq_len, 1)
#         for bif_layer in self.bifurcation_layers:
#             h = bif_layer(h, t)
        
#         # Generate output
#         output = self.output_projection(h)
        
#         return output

# class OscillatoryBifurcationDiscriminator(BaseDiscriminator):
#     """Oscillatory BifurcationGAN Discriminator"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
#         self.hidden_dim = config.DISCRIMINATOR_HIDDEN
        
#         # Feature extraction
#         self.feature_extractor = nn.Sequential(
#             nn.Conv1d(n_features, self.hidden_dim // 2, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.AdaptiveAvgPool1d(1)
#         )
        
#         # Temporal analysis
#         self.lstm = nn.LSTM(
#             input_size=n_features,
#             hidden_size=self.hidden_dim // 2,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         # Oscillation analyzer
#         self.oscillation_analyzer = nn.Sequential(
#             nn.Conv1d(n_features, self.hidden_dim // 4, kernel_size=5, padding=2),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=5, padding=2),
#             nn.LeakyReLU(0.2),
#             nn.AdaptiveAvgPool1d(1)
#         )
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(self.hidden_dim, 1)
#         )
        
#         if config.GRADIENT_PENALTY_TYPE == "wgan-gp":
#             self.classifier = nn.Sequential(
#                 nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#                 nn.LeakyReLU(0.2),
#                 nn.Linear(self.hidden_dim, 1)
#             )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size = x.size(0)
        
#         # Feature extraction
#         x_conv = x.transpose(1, 2)
#         features_conv = self.feature_extractor(x_conv).squeeze(-1)
        
#         # Temporal features
#         lstm_out, _ = self.lstm(x)
#         features_lstm = lstm_out[:, -1, :]
        
#         # Oscillation features
#         features_osc = self.oscillation_analyzer(x_conv).squeeze(-1)
        
#         # Combine
#         combined = torch.cat([features_conv, features_lstm + features_osc], dim=-1)
        
#         # Classify
#         validity = self.classifier(combined)
        
#         return validity

# # ===================== VANILLA GAN =====================

# class VanillaGenerator(BaseGenerator):
#     """Vanilla GAN Generator"""
    
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__(latent_dim, n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, seq_len * n_features),
#             nn.Tanh()
#         )
    
#     def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         seq_len = seq_len or self.seq_len
#         batch_size = z.size(0)
        
#         output = self.model(z)
#         output = output.view(batch_size, seq_len, self.n_features)
        
#         return output

# class VanillaDiscriminator(BaseDiscriminator):
#     """Vanilla GAN Discriminator"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(seq_len * n_features, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, n_features = x.shape
#         x_flat = x.view(batch_size, -1)
#         return self.model(x_flat)

# # ===================== WGAN-GP =====================

# class WGANGenerator(BaseGenerator):
#     """WGAN-GP Generator"""
    
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__(latent_dim, n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, seq_len * n_features),
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         seq_len = seq_len or self.seq_len
#         batch_size = z.size(0)
        
#         output = self.model(z)
#         output = output.view(batch_size, seq_len, self.n_features)
        
#         return output

# class WGANDiscriminator(BaseDiscriminator):
#     """WGAN-GP Critic"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(seq_len * n_features, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1)
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, n_features = x.shape
#         x_flat = x.view(batch_size, -1)
#         return self.model(x_flat)

# # ===================== TTS-GAN =====================

# class TTSGenerator(BaseGenerator):
#     """Time Series Synthesis GAN Generator"""
    
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__(latent_dim, n_features, seq_len)
#         self.hidden_dim = config.GENERATOR_HIDDEN
        
#         self.lstm = nn.LSTM(
#             input_size=latent_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=3,
#             batch_first=True,
#             dropout=0.2
#         )
        
#         self.output_projection = nn.Sequential(
#             nn.Linear(self.hidden_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_features),
#             nn.Tanh()
#         )
        
#         self.h0_proj = nn.Linear(latent_dim, self.hidden_dim)
#         self.c0_proj = nn.Linear(latent_dim, self.hidden_dim)
    
#     def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         seq_len = seq_len or self.seq_len
#         batch_size = z.size(0)
        
#         z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        
#         h0 = self.h0_proj(z).unsqueeze(0).repeat(3, 1, 1)
#         c0 = self.c0_proj(z).unsqueeze(0).repeat(3, 1, 1)
        
#         lstm_out, _ = self.lstm(z_expanded, (h0, c0))
#         output = self.output_projection(lstm_out)
        
#         return output

# class TTSDiscriminator(BaseDiscriminator):
#     """TTS-GAN Discriminator"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
#         self.hidden_dim = config.DISCRIMINATOR_HIDDEN
        
#         self.lstm = nn.LSTM(
#             input_size=n_features,
#             hidden_size=self.hidden_dim,
#             num_layers=2,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         self.output_layers = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, 128),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         lstm_out, _ = self.lstm(x)
#         last_hidden = lstm_out[:, -1, :]
#         return self.output_layers(last_hidden)

# # ===================== SigWGAN =====================

# class SignatureTransform(nn.Module):
#     """Signature transform for time series"""
    
#     def __init__(self, depth: int = 3):
#         super().__init__()
#         self.depth = depth
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, n_features = x.shape
        
#         # Add time dimension
#         time = torch.linspace(0, 1, seq_len, device=x.device)
#         time = time.view(1, seq_len, 1).expand(batch_size, -1, -1)
#         x_aug = torch.cat([time, x], dim=-1)
        
#         # Level 1: increments
#         increments = x_aug[:, 1:, :] - x_aug[:, :-1, :]
#         sig = [increments.mean(dim=1)]
        
#         # Level 2: second order
#         if self.depth >= 2:
#             for i in range(n_features + 1):
#                 for j in range(n_features + 1):
#                     if i <= j:
#                         term = x_aug[:, :, i] * x_aug[:, :, j]
#                         sig.append(term.mean(dim=1, keepdim=True))
        
#         return torch.cat(sig, dim=-1)

# class SigWGANGenerator(BaseGenerator):
#     """Signature WGAN Generator"""
    
#     def __init__(self, latent_dim: int, n_features: int, seq_len: int):
#         super().__init__(latent_dim, n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, seq_len * n_features),
#         )
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
#         seq_len = seq_len or self.seq_len
#         batch_size = z.size(0)
        
#         output = self.model(z)
#         output = output.view(batch_size, seq_len, self.n_features)
        
#         return output

# class SigWGANDiscriminator(BaseDiscriminator):
#     """Signature WGAN Discriminator"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
        
#         self.signature = SignatureTransform(depth=3)
#         sig_dim = self._get_sig_dim()
        
#         self.model = nn.Sequential(
#             nn.Linear(sig_dim, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 1)
#         )
    
#     def _get_sig_dim(self):
#         with torch.no_grad():
#             dummy = torch.zeros(1, self.seq_len, self.n_features)
#             sig = self.signature(dummy)
#             return sig.size(1)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         sig = self.signature(x)
#         return self.model(sig)

# # ===================== sisVAE (Simplified VAE) =====================

# class sisVAEEncoder(nn.Module):
#     """VAE Encoder"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__()
#         self.input_dim = seq_len * n_features
        
#         self.encoder = nn.Sequential(
#             nn.Linear(self.input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )
        
#         self.mu = nn.Linear(256, config.VAE_LATENT_DIM)
#         self.logvar = nn.Linear(256, config.VAE_LATENT_DIM)
    
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len, n_features = x.shape
#         x_flat = x.view(batch_size, -1)
        
#         h = self.encoder(x_flat)
#         mu = self.mu(h)
#         logvar = self.logvar(h)
        
#         return mu, logvar

# class sisVAEDecoder(nn.Module):
#     """VAE Decoder"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__()
#         self.seq_len = seq_len
#         self.n_features = n_features
        
#         self.decoder = nn.Sequential(
#             nn.Linear(config.VAE_LATENT_DIM, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, seq_len * n_features),
#             nn.Tanh()
#         )
    
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         batch_size = z.size(0)
#         h = self.decoder(z)
#         return h.view(batch_size, self.seq_len, self.n_features)

# class sisVAE(nn.Module):
#     """Simplified VAE model"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__()
#         self.encoder = sisVAEEncoder(n_features, seq_len)
#         self.decoder = sisVAEDecoder(n_features, seq_len)
#         self.latent_dim = config.VAE_LATENT_DIM
    
#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decoder(z)
#         return recon, mu, logvar
    
#     def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         z = torch.randn(batch_size, self.latent_dim, device=device)
#         return self.decoder(z)

# # ===================== VAE-GAN =====================

# class VAEGANGenerator(nn.Module):
#     """VAE-GAN Generator (combined VAE and GAN)"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__()
#         self.latent_dim = config.VAE_LATENT_DIM
        
#         # Encoder (for VAE part)
#         self.encoder = nn.Sequential(
#             nn.Linear(seq_len * n_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.latent_dim * 2)  # mu and logvar
#         )
        
#         # Decoder (shared)
#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, seq_len * n_features),
#             nn.Tanh()
#         )
        
#         self.seq_len = seq_len
#         self.n_features = n_features
    
#     def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len, n_features = x.shape
#         x_flat = x.view(batch_size, -1)
#         h = self.encoder(x_flat)
#         mu, logvar = h.chunk(2, dim=-1)
#         return mu, logvar
    
#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z: torch.Tensor) -> torch.Tensor:
#         batch_size = z.size(0)
#         h = self.decoder(z)
#         return h.view(batch_size, self.seq_len, self.n_features)
    
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar, z
    
#     def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         z = torch.randn(batch_size, self.latent_dim, device=device)
#         return self.decode(z)

# class VAEGANDiscriminator(BaseDiscriminator):
#     """VAE-GAN Discriminator (shared with Vanilla GAN)"""
    
#     def __init__(self, n_features: int, seq_len: int):
#         super().__init__(n_features, seq_len)
        
#         self.model = nn.Sequential(
#             nn.Linear(seq_len * n_features, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, n_features = x.shape
#         x_flat = x.view(batch_size, -1)
#         return self.model(x_flat)

# # ===================== MODEL FACTORY =====================

# def create_model(model_type: str, n_features: int, seq_len: int, device: torch.device):
#     """Factory function to create generator and discriminator"""
    
#     if model_type == "oscillatory_bifurcation_gan":
#         generator = OscillatoryBifurcationGenerator(config.LATENT_DIM, n_features, seq_len)
#         discriminator = OscillatoryBifurcationDiscriminator(n_features, seq_len)
    
#     elif model_type == "vanilla_gan":
#         generator = VanillaGenerator(config.LATENT_DIM, n_features, seq_len)
#         discriminator = VanillaDiscriminator(n_features, seq_len)
    
#     elif model_type == "wgan_gp":
#         generator = WGANGenerator(config.LATENT_DIM, n_features, seq_len)
#         discriminator = WGANDiscriminator(n_features, seq_len)
    
#     elif model_type == "tts_gan":
#         generator = TTSGenerator(config.LATENT_DIM, n_features, seq_len)
#         discriminator = TTSDiscriminator(n_features, seq_len)
    
#     elif model_type == "sig_wgan":
#         generator = SigWGANGenerator(config.LATENT_DIM, n_features, seq_len)
#         discriminator = SigWGANDiscriminator(n_features, seq_len)
    
#     elif model_type == "sisvae":
#         # For VAE, we return the VAE model as generator, and a separate discriminator isn't needed
#         generator = sisVAE(n_features, seq_len)
#         discriminator = None
    
#     elif model_type == "vae_gan":
#         generator = VAEGANGenerator(n_features, seq_len)
#         discriminator = VAEGANDiscriminator(n_features, seq_len)
    
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
    
#     # Move to device
#     if generator is not None:
#         generator = generator.to(device)
#     if discriminator is not None:
#         discriminator = discriminator.to(device)
    
#     return generator, discriminator

# def count_parameters(model: nn.Module) -> int:
#     """Count trainable parameters"""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from config import config

# ===================== BASE MODELS =====================

class BaseGenerator(nn.Module):
    """Base generator class"""
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.seq_len = seq_len

class BaseDiscriminator(nn.Module):
    """Base discriminator class"""
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len

# ===================== OSCILLATORY BIFURCATION GAN =====================

class OscillatoryBifurcationLayer(nn.Module):
    """Oscillatory bifurcation dynamics layer - NO IN-PLACE OPERATIONS"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_oscillators = min(config.N_OSCILLATORS, max(1, hidden_dim // 8))
        
        # Bifurcation parameters (learnable)
        self.mu = nn.Parameter(torch.tensor(config.HOPF_MU))
        self.omega = nn.Parameter(torch.tensor(config.HOPF_OMEGA))
        self.alpha = nn.Parameter(torch.tensor(config.HOPF_ALPHA))
        
        # Oscillator parameters (learnable)
        self.frequencies = nn.Parameter(torch.tensor(config.NATURAL_FREQUENCIES[:self.n_oscillators]))
        
        # Coupling matrix (learnable)
        self.coupling = nn.Parameter(torch.randn(self.n_oscillators, self.n_oscillators) * config.OSCILLATOR_COUPLING)
        
        # Transformations
        self.to_oscillator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.n_oscillators * 2)
        )
        
        self.from_oscillator = nn.Sequential(
            nn.Linear(self.n_oscillators * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Time modulation
        self.time_mod = nn.Linear(1, self.n_oscillators)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Project to oscillator space
        osc_state = self.to_oscillator(x)  # (batch, seq_len, n_oscillators*2)
        
        # Split into phase and amplitude
        phase = osc_state[:, :, :self.n_oscillators]  # (batch, seq_len, n_oscillators)
        amplitude = osc_state[:, :, self.n_oscillators:]  # (batch, seq_len, n_oscillators)
        
        # Time modulation
        t_mod = self.time_mod(t)  # (batch, seq_len, n_oscillators)
        
        # Create output tensors (avoid in-place operations)
        new_phase = []
        new_amplitude = []
        
        # Process each time step
        for i in range(seq_len):
            if i == 0:
                # First time step - use as is
                new_phase.append(phase[:, i:i+1, :])
                new_amplitude.append(amplitude[:, i:i+1, :])
            else:
                # Previous values
                phase_prev = phase[:, i-1:i, :]  # (batch, 1, n_oscillators)
                amp_prev = amplitude[:, i-1:i, :]  # (batch, 1, n_oscillators)
                
                # Phase dynamics - compute without in-place
                # Expand for coupling calculation
                phase_prev_exp = phase_prev.unsqueeze(-1)  # (batch, 1, n_oscillators, 1)
                phase_prev_exp_t = phase_prev.unsqueeze(-2)  # (batch, 1, 1, n_oscillators)
                
                # Phase differences
                phase_diff = phase_prev_exp - phase_prev_exp_t  # (batch, 1, n_oscillators, n_oscillators)
                
                # Coupling term
                coupling_term = torch.matmul(
                    torch.sin(phase_diff), 
                    self.coupling.unsqueeze(0).unsqueeze(0)
                )  # (batch, 1, n_oscillators, n_oscillators)
                
                # Sum over oscillators
                coupling_sum = coupling_term.sum(dim=-1)  # (batch, 1, n_oscillators)
                
                # Phase update
                dphase = self.frequencies.unsqueeze(0).unsqueeze(0) + coupling_sum * config.OSCILLATOR_COUPLING
                
                # Add time modulation
                dphase = dphase + t_mod[:, i:i+1, :] * 0.1
                
                # Add noise if training
                if config.PHASE_NOISE_STD > 0 and self.training:
                    noise = torch.randn_like(dphase) * config.PHASE_NOISE_STD
                    dphase = dphase + noise
                
                # New phase (no in-place)
                phase_new = phase_prev + dphase * 0.1
                
                # Amplitude dynamics
                # Hopf bifurcation: dr/dt = μr - r³
                damp = -config.AMPLITUDE_DECAY * amp_prev
                damp = damp + self.mu * amp_prev - amp_prev**3
                
                # New amplitude (no in-place)
                amp_new = amp_prev + damp * 0.1
                
                # Saturate amplitude
                amp_new = torch.tanh(amp_new) * config.AMPLITUDE_SATURATION
                
                new_phase.append(phase_new)
                new_amplitude.append(amp_new)
        
        # Concatenate along sequence dimension
        phase_out = torch.cat(new_phase, dim=1)  # (batch, seq_len, n_oscillators)
        amp_out = torch.cat(new_amplitude, dim=1)  # (batch, seq_len, n_oscillators)
        
        # Combine and project back
        combined = torch.cat([phase_out, amp_out], dim=-1)  # (batch, seq_len, n_oscillators*2)
        oscillator_output = self.from_oscillator(combined)  # (batch, seq_len, hidden_dim)
        
        # Residual connection (no in-place)
        output = x + oscillator_output * 0.1
        
        return output

class OscillatoryBifurcationGenerator(BaseGenerator):
    """Oscillatory BifurcationGAN Generator - WITH PROPER GRADIENT FLOW"""
    
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__(latent_dim, n_features, seq_len)
        self.hidden_dim = config.GENERATOR_HIDDEN
        
        # Noise processing
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim * 4),
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
        
        # Temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.InstanceNorm1d(self.hidden_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(2)
        ])
        
        # Oscillatory bifurcation layers
        self.bifurcation_layers = nn.ModuleList([
            OscillatoryBifurcationLayer(self.hidden_dim)
            for _ in range(2)
        ])
        
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _create_pos_encoding(self, seq_len):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           -(np.log(10000.0) / self.hidden_dim))
        pe = torch.zeros(seq_len, self.hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        batch_size = z.size(0)
        seq_len = seq_len or self.seq_len
        
        # Process noise
        h = self.noise_processor(z)  # (batch, hidden_dim)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # Use expand instead of repeat
        
        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]
        
        # Apply temporal convolutions
        h_t = h.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        for conv in self.temporal_convs:
            h_conv = conv(h_t)
            h_t = h_t + h_conv * 0.1
        h = h_t.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Apply oscillatory bifurcation dynamics
        t = torch.arange(seq_len, device=z.device).float().view(1, seq_len, 1)
        for bif_layer in self.bifurcation_layers:
            h = bif_layer(h, t)
        
        # Generate output
        output = self.output_projection(h)  # (batch, seq_len, n_features)
        
        return output

class OscillatoryBifurcationDiscriminator(BaseDiscriminator):
    """Oscillatory BifurcationGAN Discriminator - FIXED"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        self.hidden_dim = config.DISCRIMINATOR_HIDDEN
        
        # Feature extraction - using Conv1d
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, self.hidden_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Temporal analysis
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),  # conv_features + lstm_features
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1)
        )
        
        if config.GRADIENT_PENALTY_TYPE == "wgan-gp":
            # Remove sigmoid for WGAN-GP
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, 1)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Conv features
        x_conv = x.transpose(1, 2)  # (batch, n_features, seq_len)
        conv_out = self.conv_layers(x_conv)  # (batch, hidden_dim, seq_len)
        conv_features = self.global_pool(conv_out).squeeze(-1)  # (batch, hidden_dim)
        
        # LSTM features
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        lstm_features = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Combine features
        combined = torch.cat([conv_features, lstm_features], dim=-1)  # (batch, hidden_dim*2)
        
        # Classify
        validity = self.classifier(combined)
        
        return validity

# ===================== VANILLA GAN =====================

class VanillaGenerator(BaseGenerator):
    """Vanilla GAN Generator"""
    
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__(latent_dim, n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, seq_len * n_features),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, self.n_features)
        
        return output

class VanillaDiscriminator(BaseDiscriminator):
    """Vanilla GAN Discriminator"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(seq_len * n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        x_flat = x.reshape(batch_size, -1)  # Use reshape instead of view
        return self.model(x_flat)

# ===================== WGAN-GP =====================

class WGANGenerator(BaseGenerator):
    """WGAN-GP Generator"""
    
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__(latent_dim, n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, seq_len * n_features),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, self.n_features)
        
        return output

class WGANDiscriminator(BaseDiscriminator):
    """WGAN-GP Critic"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(seq_len * n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        x_flat = x.reshape(batch_size, -1)  # Use reshape instead of view
        return self.model(x_flat)

# ===================== TTS-GAN =====================

class TTSGenerator(BaseGenerator):
    """Time Series Synthesis GAN Generator"""
    
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__(latent_dim, n_features, seq_len)
        self.hidden_dim = config.GENERATOR_HIDDEN
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_features),
            nn.Tanh()
        )
        
        self.h0_proj = nn.Linear(latent_dim, self.hidden_dim)
        self.c0_proj = nn.Linear(latent_dim, self.hidden_dim)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        h0 = self.h0_proj(z).unsqueeze(0).repeat(3, 1, 1)
        c0 = self.c0_proj(z).unsqueeze(0).repeat(3, 1, 1)
        
        lstm_out, _ = self.lstm(z_expanded, (h0, c0))
        output = self.output_projection(lstm_out)
        
        return output

class TTSDiscriminator(BaseDiscriminator):
    """TTS-GAN Discriminator"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        self.hidden_dim = config.DISCRIMINATOR_HIDDEN
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.output_layers(last_hidden)

# ===================== SigWGAN =====================

class SignatureTransform(nn.Module):
    """Signature transform for time series"""
    
    def __init__(self, depth: int = 3):
        super().__init__()
        self.depth = depth
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        
        # Add time dimension
        time = torch.linspace(0, 1, seq_len, device=x.device)
        time = time.view(1, seq_len, 1).expand(batch_size, -1, -1)
        x_aug = torch.cat([time, x], dim=-1)
        aug_features = n_features + 1
        
        # Level 1: increments
        increments = x_aug[:, 1:, :] - x_aug[:, :-1, :]
        sig = [increments.mean(dim=1)]
        
        # Level 2: second order
        if self.depth >= 2:
            for i in range(aug_features):
                for j in range(aug_features):
                    if i <= j:
                        term = x_aug[:, :, i] * x_aug[:, :, j]
                        sig.append(term.mean(dim=1, keepdim=True))
        
        return torch.cat(sig, dim=-1)

class SigWGANGenerator(BaseGenerator):
    """Signature WGAN Generator"""
    
    def __init__(self, latent_dim: int, n_features: int, seq_len: int):
        super().__init__(latent_dim, n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, seq_len * n_features),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, self.n_features)
        
        return output

class SigWGANDiscriminator(BaseDiscriminator):
    """Signature WGAN Discriminator - FIXED VERSION"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        
        self.signature = SignatureTransform(depth=3)
        sig_dim = self._get_sig_dim(n_features)
        
        self.model = nn.Sequential(
            nn.Linear(sig_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def _get_sig_dim(self, n_features):
        """Calculate signature dimension without creating tensors"""
        # Approximate dimension: (n_features+1) * (n_features+2) // 2
        return (n_features + 1) * (n_features + 2) // 2 + n_features + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensor is contiguous before using in signature transform
        x = x.contiguous()
        sig = self.signature(x)
        return self.model(sig)

# ===================== sisVAE (Simplified VAE) =====================

class sisVAEEncoder(nn.Module):
    """VAE Encoder"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.input_dim = seq_len * n_features
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(256, config.VAE_LATENT_DIM)
        self.logvar = nn.Linear(256, config.VAE_LATENT_DIM)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_features = x.shape
        x_flat = x.reshape(batch_size, -1)  # Use reshape instead of view
        
        h = self.encoder(x_flat)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        return mu, logvar

class sisVAEDecoder(nn.Module):
    """VAE Decoder"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        
        self.decoder = nn.Sequential(
            nn.Linear(config.VAE_LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * n_features),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        h = self.decoder(z)
        return h.reshape(batch_size, self.seq_len, self.n_features)  # Use reshape

class sisVAE(nn.Module):
    """Simplified VAE model"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.encoder = sisVAEEncoder(n_features, seq_len)
        self.decoder = sisVAEDecoder(n_features, seq_len)
        self.latent_dim = config.VAE_LATENT_DIM
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decoder(z)

# ===================== VAE-GAN =====================

class VAEGANGenerator(nn.Module):
    """VAE-GAN Generator (combined VAE and GAN)"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()
        self.latent_dim = config.VAE_LATENT_DIM
        self.seq_len = seq_len
        self.n_features = n_features
        
        # Encoder (for VAE part)
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2)  # mu and logvar
        )
        
        # Decoder (shared)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * n_features),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_features = x.shape
        x_flat = x.reshape(batch_size, -1)  # Use reshape
        h = self.encoder(x_flat)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        h = self.decoder(z)
        return h.reshape(batch_size, self.seq_len, self.n_features)  # Use reshape
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)

class VAEGANDiscriminator(BaseDiscriminator):
    """VAE-GAN Discriminator (shared with Vanilla GAN)"""
    
    def __init__(self, n_features: int, seq_len: int):
        super().__init__(n_features, seq_len)
        
        self.model = nn.Sequential(
            nn.Linear(seq_len * n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        x_flat = x.reshape(batch_size, -1)  # Use reshape
        return self.model(x_flat)

# ===================== MODEL FACTORY =====================

def create_model(model_type: str, n_features: int, seq_len: int, device: torch.device):
    """Factory function to create generator and discriminator"""
    
    if model_type == "oscillatory_bifurcation_gan":
        generator = OscillatoryBifurcationGenerator(config.LATENT_DIM, n_features, seq_len)
        discriminator = OscillatoryBifurcationDiscriminator(n_features, seq_len)
    
    elif model_type == "vanilla_gan":
        generator = VanillaGenerator(config.LATENT_DIM, n_features, seq_len)
        discriminator = VanillaDiscriminator(n_features, seq_len)
    
    elif model_type == "wgan_gp":
        generator = WGANGenerator(config.LATENT_DIM, n_features, seq_len)
        discriminator = WGANDiscriminator(n_features, seq_len)
    
    elif model_type == "tts_gan":
        generator = TTSGenerator(config.LATENT_DIM, n_features, seq_len)
        discriminator = TTSDiscriminator(n_features, seq_len)
    
    elif model_type == "sig_wgan":
        generator = SigWGANGenerator(config.LATENT_DIM, n_features, seq_len)
        discriminator = SigWGANDiscriminator(n_features, seq_len)
    
    elif model_type == "sisvae":
        # For VAE, we return the VAE model as generator, and a separate discriminator isn't needed
        generator = sisVAE(n_features, seq_len)
        discriminator = None
    
    elif model_type == "vae_gan":
        generator = VAEGANGenerator(n_features, seq_len)
        discriminator = VAEGANDiscriminator(n_features, seq_len)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device
    if generator is not None:
        generator = generator.to(device)
    if discriminator is not None:
        discriminator = discriminator.to(device)
    
    return generator, discriminator

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)