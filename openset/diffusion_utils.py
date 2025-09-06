# diffusion_utils.py
import torch
import torch.nn.functional as F

# Define variance schedule
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# --- Pre-calculate diffusion constants ---
TIMESTEPS = 200 # Number of diffusion steps
betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# --- Helper functions ---

def q_sample(x_start, t, noise=None):
    """Forward diffusion process: add noise to data"""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Extract the correct values for the batch
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod.to(x_start.device)[t, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.to(x_start.device)[t, None]
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, t, class_labels):
    """Denoise one step"""
    betas_t = betas.to(x.device)[t, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.to(x.device)[t, None]
    sqrt_recip_alphas_t = sqrt_recip_alphas.to(x.device)[t, None]
    
    # Equation 11 in the DDPM paper
    # Use our model (noise predictor) to predict the mean
    predicted_noise = model(x, t, class_labels)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t == 0:
        return model_mean
    else:
        posterior_variance_t = posterior_variance.to(x.device)[t, None]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def denoise_feature(model, feature, condition_class, timesteps=TIMESTEPS):
    """Full denoising process for a single feature vector, conditioned on a class"""
    # Start from pure noise
    device = feature.device
    z_noisy = torch.randn_like(feature, device=device)
    
    # Mildly noise the original feature to guide the process
    t_start = 50 # Start denoising from a partially noised state
    z_t = q_sample(feature, torch.full((feature.shape[0],), t_start, device=device, dtype=torch.long))

    # Denoise step-by-step
    for i in reversed(range(0, t_start)):
        t = torch.full((feature.shape[0],), i, device=device, dtype=torch.long)
        class_labels = torch.full((feature.shape[0],), condition_class, device=device, dtype=torch.long)
        z_t = p_sample(model.diffusion_model, z_t, t, class_labels)
        
    return z_t