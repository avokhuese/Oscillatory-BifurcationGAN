# metrics.py
"""
Evaluation metrics for GANs and time series generation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class InceptionScore:
    """Inception Score for time series (adapted)"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def calculate(self, real_data, fake_data, splits=10):
        """
        Calculate Inception Score
        Args:
            real_data: [n_samples, seq_len, n_features]
            fake_data: [n_samples, seq_len, n_features]
        """
        # Compute features (use mean and std along sequence)
        real_features = torch.cat([
            real_data.mean(dim=1),
            real_data.std(dim=1)
        ], dim=1)
        
        fake_features = torch.cat([
            fake_data.mean(dim=1),
            fake_data.std(dim=1)
        ], dim=1)
        
        # Compute conditional probabilities
        scores = []
        for i in range(0, len(fake_features), len(fake_features) // splits):
            split = fake_features[i:i + len(fake_features) // splits]
            
            # Compute KL divergence
            p = F.softmax(split, dim=1)
            q = F.softmax(real_features, dim=1).mean(dim=0, keepdim=True)
            
            kl = (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=1)
            scores.append(torch.exp(kl.mean()))
        
        return torch.tensor(scores).mean().item(), torch.tensor(scores).std().item()

class FrechetInceptionDistance:
    """Frechet Inception Distance for time series"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def calculate(self, real_data, fake_data):
        """
        Calculate FID between real and fake data
        """
        # Compute features (use mean and std along sequence)
        real_features = torch.cat([
            real_data.mean(dim=1),
            real_data.std(dim=1)
        ], dim=1).cpu().numpy()
        
        fake_features = torch.cat([
            fake_data.mean(dim=1),
            fake_data.std(dim=1)
        ], dim=1).cpu().numpy()
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical error might give imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid

class TimeSeriesMetrics:
    """Time-series specific evaluation metrics"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_basic_metrics(self, real, pred):
        """
        Calculate basic regression metrics
        Args:
            real: [n_samples, seq_len, n_features]
            pred: [n_samples, seq_len, n_features]
        """
        real_flat = real.reshape(-1, real.shape[-1])
        pred_flat = pred.reshape(-1, pred.shape[-1])
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['MAE'] = mean_absolute_error(real_flat, pred_flat)
        
        # Mean Squared Error
        metrics['MSE'] = mean_squared_error(real_flat, pred_flat)
        
        # Root Mean Squared Error
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        
        # Median Absolute Error
        metrics['MedAE'] = np.median(np.abs(real_flat - pred_flat))
        
        return metrics
    
    def autocorrelation_similarity(self, real, fake, max_lag=20):
        """
        Compare autocorrelation functions
        """
        def autocorr(x):
            x = x - x.mean()
            result = np.correlate(x, x, mode='full')
            result = result[result.size // 2:]
            result = result / result[0]
            return result[:max_lag]
        
        real_flat = real.reshape(-1, real.shape[-1])
        fake_flat = fake.reshape(-1, fake.shape[-1])
        
        real_acf = np.array([autocorr(real_flat[:, i]) for i in range(real_flat.shape[1])])
        fake_acf = np.array([autocorr(fake_flat[:, i]) for i in range(fake_flat.shape[1])])
        
        # Cosine similarity of ACFs
        similarity = np.mean([
            np.dot(real_acf[i], fake_acf[i]) / 
            (np.linalg.norm(real_acf[i]) * np.linalg.norm(fake_acf[i]) + 1e-8)
            for i in range(real_flat.shape[1])
        ])
        
        return similarity
    
    def spectral_similarity(self, real, fake):
        """
        Compare power spectral density
        """
        def spectrum(x):
            fft = np.fft.fft(x, axis=0)
            power = np.abs(fft) ** 2
            return power[:len(power)//2]
        
        real_spec = spectrum(real.reshape(-1, real.shape[-1]))
        fake_spec = spectrum(fake.reshape(-1, fake.shape[-1]))
        
        # Normalize
        real_spec = real_spec / (real_spec.sum() + 1e-8)
        fake_spec = fake_spec / (fake_spec.sum() + 1e-8)
        
        # KL divergence
        kl_div = (real_spec * np.log(real_spec / (fake_spec + 1e-8) + 1e-8)).sum()
        
        return kl_div
    
    def discriminative_score(self, real, fake, test_size=0.2):
        """
        Train a classifier to distinguish real vs fake
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare data
        real_features = real.reshape(real.shape[0], -1)
        fake_features = fake.reshape(fake.shape[0], -1)
        
        X = np.vstack([real_features, fake_features])
        y = np.hstack([np.ones(len(real_features)), np.zeros(len(fake_features))])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        score = clf.score(X_test, y_test)
        
        # Return 1 - accuracy (lower is better, 0.5 is random)
        return max(0, score - 0.5) * 2
    
    def predictive_score(self, train_real, test_real, train_fake, test_fake):
        """
        Train on generated data, test on real data (TSTR)
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Reshape data
        train_real_flat = train_real.reshape(train_real.shape[0], -1)
        train_fake_flat = train_fake.reshape(train_fake.shape[0], -1)
        test_real_flat = test_real.reshape(test_real.shape[0], -1)
        
        # Train on real data
        model_real = RandomForestRegressor(n_estimators=100, random_state=42)
        model_real.fit(train_real_flat, np.zeros(len(train_real_flat)))  # Dummy target
        
        # Train on fake data
        model_fake = RandomForestRegressor(n_estimators=100, random_state=42)
        model_fake.fit(train_fake_flat, np.zeros(len(train_fake_flat)))
        
        # Compare predictions (use reconstruction error)
        pred_real = model_real.predict(test_real_flat)
        pred_fake = model_fake.predict(test_real_flat)
        
        # MSE between predictions (lower means fake data better represents real)
        mse = mean_squared_error(pred_real, pred_fake)
        
        return mse
    
    def wasserstein_distance(self, real, fake):
        """
        Calculate Wasserstein distance between distributions
        """
        real_flat = real.reshape(-1, real.shape[-1])
        fake_flat = fake.reshape(-1, fake.shape[-1])
        
        distances = []
        for i in range(real_flat.shape[1]):
            dist = wasserstein_distance(real_flat[:, i], fake_flat[:, i])
            distances.append(dist)
        
        return np.mean(distances)

class GANLosses:
    """GAN-specific losses for monitoring"""
    
    @staticmethod
    def wasserstein_loss(real_validity, fake_validity):
        """Wasserstein loss"""
        return -torch.mean(real_validity) + torch.mean(fake_validity)
    
    @staticmethod
    def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10):
        """Gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1).to(real_data.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        # Get discriminator output
        d_interpolates = discriminator(interpolates)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    @staticmethod
    def bifurcation_loss(generator, real_data, lambda_bif=0.1):
        """
        Bifurcation regularization loss
        Encourages oscillatory behavior
        """
        # Generate fake data
        batch_size = real_data.size(0)
        z = torch.randn(batch_size, generator.latent_dim).to(real_data.device)
        fake_data = generator(z)
        
        # Compute bifurcation features
        real_fft = torch.fft.fft(real_data, dim=1)
        fake_fft = torch.fft.fft(fake_data, dim=1)
        
        # Frequency domain loss
        freq_loss = F.mse_loss(torch.abs(real_fft), torch.abs(fake_fft))
        
        # Autocorrelation loss
        real_acf = torch.stack([self._autocorr(real_data[:, :, i]) for i in range(real_data.shape[-1])], dim=-1)
        fake_acf = torch.stack([self._autocorr(fake_data[:, :, i]) for i in range(fake_data.shape[-1])], dim=-1)
        acf_loss = F.mse_loss(real_acf, fake_acf)
        
        return lambda_bif * (freq_loss + acf_loss)
    
    @staticmethod
    def _autocorr(x, max_lag=10):
        """Compute autocorrelation"""
        x = x - x.mean(dim=1, keepdim=True)
        x = F.pad(x, (0, 0, 0, max_lag))
        
        result = []
        for lag in range(max_lag):
            corr = (x[:, :-lag-1] * x[:, lag+1:]).mean(dim=1)
            result.append(corr)
        
        return torch.stack(result, dim=1)