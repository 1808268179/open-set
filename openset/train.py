# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from diffusion_utils import TIMESTEPS, q_sample

# 假设您原来的损失函数也在这里
from models import compute_enhanced_adversarial_loss 

def train_fe_stage(num_epochs, model, train_loader, device):
    """Stage 1: Train the DCLNTSHybridNet feature extractor."""
    print("--- Stage 1: Training Feature Extractor ---")
    model.to(device)
    model.freeze_feature_extractor(freeze=False) # 确保模型可训练

    # --- 设置您原来的优化器 ---
    # (这里是您原来代码的直接应用)
    backbone_params = list(model.feature_extractor.backbone.parameters())
    # ... (其他所有部分的参数)
    noise_detector_params = list(model.feature_extractor.noise_detector.parameters())
    discriminator_params = list(model.feature_extractor.discriminator.parameters())
    ssl_params = list(model.feature_extractor.contrastive_projection.parameters())
    
    optimizer_backbone = optim.SGD(backbone_params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    # ... (所有其他的优化器)
    optimizer_noise = optim.SGD(noise_detector_params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer_disc = optim.SGD(discriminator_params, lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer_ssl = optim.SGD(ssl_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

    scheduler_backbone = optim.lr_scheduler.MultiStepLR(optimizer_backbone, milestones=[50, 80], gamma=0.1)
    scheduler_noise = optim.lr_scheduler.MultiStepLR(optimizer_noise, milestones=[50, 80], gamma=0.1)
    scheduler_disc = optim.lr_scheduler.MultiStepLR(optimizer_disc, milestones=[50, 80], gamma=0.1)
    scheduler_ssl = optim.lr_scheduler.MultiStepLR(optimizer_ssl, milestones=[50, 80], gamma=0.1)

    for epoch in range(1, num_epochs + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer_backbone.zero_grad(); #... .zero_grad() for all
            optimizer_noise.zero_grad()
            optimizer_disc.zero_grad()
            optimizer_ssl.zero_grad()
            
            # 这里调用的是您 DCLNTSHybridNet 的 forward
            outputs = model.feature_extractor(inputs) 
            # 假设您原来的损失函数叫这个名字
            loss_dict = compute_enhanced_adversarial_loss(outputs, targets)
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            
            optimizer_backbone.step(); # ... .step() for all
            optimizer_noise.step()
            optimizer_disc.step()
            optimizer_ssl.step()

            progress_bar.set_postfix(loss=f"{total_loss.item():.3f}")

    print("--- Feature Extractor Training Finished ---")
    torch.save(model.state_dict(), 'openset_model_fe_trained.pth')

def train_dm_stage(num_epochs, model, train_loader, device):
    """Stage 2: Train the FeatureDiffusionModel."""
    print("--- Stage 2: Training Diffusion Model ---")
    model.to(device)
    model.freeze_feature_extractor(freeze=True) # 冻结特征提取器
    
    optimizer = torch.optim.Adam(model.diffusion_model.parameters(), lr=1e-4)
    
    for epoch in range(1, num_epochs + 1):
        model.diffusion_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            features = model.get_features(images)
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=device).long()
            noise = torch.randn_like(features)
            features_noisy = q_sample(features, t, noise)
            
            predicted_noise = model.diffusion_model(features_noisy, t, labels)
            loss = F.mse_loss(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(mse_loss=f"{loss.item():.4f}")

    print("--- Diffusion Model Training Finished ---")
    torch.save(model.state_dict(), 'openset_model_dm_trained.pth')