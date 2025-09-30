import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, 
                 noise_steps, 
                 beta_start, 
                 beta_end, 
                 device,
                 sigma_min=0.01, 
                 sigma_max=1.0):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # DDPM discrete noise schedule
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # For score-based diffusion
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # ------------------------
    # DDPM Forward Process
    # ------------------------
    def forward_images(self, x, t):
        alpha_t = self.alpha_hat[t]
        mean_t = torch.sqrt(alpha_t)[:, None, None]
        var_t = torch.sqrt(1 - alpha_t)[:, None, None]
        
        epsilon = torch.randn_like(x)
        return (mean_t * x) + (var_t * epsilon), epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    # ------------------------
    # Score-based additions
    # ------------------------
    def sigma_fn(self, t):
        # Continuous noise scale function for score-based models
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sample_continuous_timesteps(self, n):
        # Sample t ∈ (0, 1) for score matching
        return torch.rand(n, device=self.device) * 0.999 + 0.001

    def forward_process_score(self, x0, t):
        # Add noise using σ(t) for score matching.
        sigma_t = self.sigma_fn(t).view(-1, *[1] * (x0.ndim - 1))  # broadcast shape
        epsilon = torch.randn_like(x0)
        x_t = x0 + sigma_t * epsilon
        return x_t, epsilon, sigma_t

    def score_matching_loss(self, score_pred, epsilon, sigma_t):
        target_score = -epsilon / sigma_t
        return F.mse_loss(score_pred, target_score)
