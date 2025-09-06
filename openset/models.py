import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable, Function
from einops import rearrange

class GradientReversal(Function):
    """梯度反转层实现"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    """梯度反转层的便捷调用函数"""
    return GradientReversal.apply(x, lambda_)

class RegionConfusionMechanism(nn.Module):
    """区域混淆机制 - DCL的破坏模块 (单一粒度版本)"""
    def __init__(self, N=4, k=2):
        super(RegionConfusionMechanism, self).__init__()
        self.N = N
        self.k = k
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        pad_h = (self.N - height % self.N) % self.N
        pad_w = (self.N - width % self.N) % self.N
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            height += pad_h
            width += pad_w
        
        region_h = height // self.N
        region_w = width // self.N
        
        x_regions = x.view(batch_size, channels, self.N, region_h, self.N, region_w)
        x_regions = x_regions.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        shuffled_regions = torch.zeros_like(x_regions)
        
        for b in range(batch_size):
            visited = np.zeros((self.N, self.N), dtype=bool)
            
            for i in range(self.N):
                for j in range(self.N):
                    if visited[i, j]:
                        continue
                    
                    candidates = []
                    for ni in range(max(0, i - self.k), min(self.N, i + self.k + 1)):
                        for nj in range(max(0, j - self.k), min(self.N, j + self.k + 1)):
                            if not visited[ni, nj]:
                                candidates.append((ni, nj))
                    
                    if candidates:
                        target_i, target_j = candidates[np.random.randint(len(candidates))]
                        shuffled_regions[b, target_i, target_j] = x_regions[b, i, j]
                        visited[target_i, target_j] = True
                    else:
                        shuffled_regions[b, i, j] = x_regions[b, i, j]
                        visited[i, j] = True
        
        shuffled_x = shuffled_regions.permute(0, 3, 1, 4, 2, 5).contiguous()
        shuffled_x = shuffled_x.view(batch_size, channels, height, width)
        
        if pad_h > 0 or pad_w > 0:
            shuffled_x = shuffled_x[:, :, :height-pad_h, :width-pad_w]
            
        return shuffled_x

class NoiseDetector(nn.Module):
    """噪声检测器"""
    def __init__(self, feature_dim=512):
        super(NoiseDetector, self).__init__()
        self.down1 = nn.Conv2d(feature_dim, 256, 3, 1, 1)
        self.down2 = nn.Conv2d(256, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 64, 3, 2, 1)
        
        self.noise_pred1 = nn.Conv2d(256, 1, 1, 1, 0)
        self.noise_pred2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.noise_pred3 = nn.Conv2d(64, 1, 1, 1, 0)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features):
        d1 = self.relu(self.down1(features))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        
        noise_map1 = self.sigmoid(self.noise_pred1(d1))
        noise_map2 = self.sigmoid(self.noise_pred2(d2))
        noise_map3 = self.sigmoid(self.noise_pred3(d3))
        
        h, w = noise_map1.size(2), noise_map1.size(3)
        noise_map2 = F.interpolate(noise_map2, size=(h, w), mode='bilinear', align_corners=False)
        noise_map3 = F.interpolate(noise_map3, size=(h, w), mode='bilinear', align_corners=False)
        
        noise_map = (noise_map1 + noise_map2 + noise_map3) / 3.0
        return noise_map

class ResNetBackbone(nn.Module):
    """主分类网络骨干"""
    def __init__(self, num_classes=200, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self.feature_dim = 512
        
    def forward(self, x, return_features=False):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        feature_map = x
        pooled = self.avgpool(x)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, feature_map
        else:
            return logits

class DCLNTSHybridNet(nn.Module):
    """DCL-NTS混合网络 (精简版)"""
    def __init__(self, num_classes=200, N=[2, 4, 7], k=2, lambda_adv=1.0, 
                 contrastive_temp=0.07, projection_dim=128):
        super(DCLNTSHybridNet, self).__init__()
        
        # 根据N列表创建多个RCM实例
        self.N_list = N if isinstance(N, list) else [N]
        self.k = k
        self.rcms = nn.ModuleList([RegionConfusionMechanism(N=n, k=self.k) for n in self.N_list])

        self.backbone = ResNetBackbone(num_classes=num_classes)
        self.noise_detector = NoiseDetector(feature_dim=512)
        
        # 1. 对比学习投影头
        self.contrastive_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )
        
        # 2. 增强的对抗判别器
        self.discriminator = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
        # === 移除了 域混淆判别器 (domain_discriminator) ===
        
        self.lambda_adv = lambda_adv
        self.contrastive_temp = contrastive_temp
        
    def forward(self, x, return_features_for_vis=False):
        batch_size = x.size(0)
        
        if self.training:
            shuffled_images = [rcm(x) for rcm in self.rcms]
            num_granularities = len(shuffled_images)

            all_images = torch.cat([x] + shuffled_images, dim=0)
            all_logits, all_features = self.backbone(all_images, return_features=True)

            logits_orig = all_logits[:batch_size]
            features_orig = all_features[:batch_size]
            
            logits_shuffled_list = list(torch.chunk(all_logits[batch_size:], num_granularities, dim=0))
            features_shuffled_list = list(torch.chunk(all_features[batch_size:], num_granularities, dim=0))
            
            noise_map_orig = self.noise_detector(features_orig)
            noise_map_shuffled_list = [self.noise_detector(f_s) for f_s in features_shuffled_list]
            
            pooled_orig = F.adaptive_avg_pool2d(features_orig, (1, 1)).view(batch_size, -1)
            pooled_shuffled_list = [F.adaptive_avg_pool2d(f_s, (1, 1)).view(batch_s.size(0), -1) for f_s, batch_s in zip(features_shuffled_list, shuffled_images)]

            proj_orig = self.contrastive_projection(pooled_orig)
            proj_shuffled_list = [self.contrastive_projection(p_s) for p_s in pooled_shuffled_list]
            
            pooled_orig_adv = grad_reverse(pooled_orig, self.lambda_adv)
            pooled_shuffled_adv_list = [grad_reverse(p_s, self.lambda_adv) for p_s in pooled_shuffled_list]
            
            disc_orig = self.discriminator(pooled_orig_adv)
            disc_shuffled_list = [self.discriminator(p_s_adv) for p_s_adv in pooled_shuffled_adv_list]
            
            # --- 返回精简后的输出字典 (移除了 domain_pred) ---
            return {
                'logits_orig': logits_orig,
                'noise_map_orig': noise_map_orig,
                'features_orig': features_orig,
                'proj_orig': proj_orig,
                'disc_orig': disc_orig,
                
                'logits_shuffled': logits_shuffled_list,
                'noise_map_shuffled': noise_map_shuffled_list,
                'features_shuffled': features_shuffled_list,
                'proj_shuffled': proj_shuffled_list,
                'disc_shuffled': disc_shuffled_list,
            }
        else: # self.training is False (evaluation or inference mode)
            # 我们简化原有逻辑，让eval模式总是返回统一格式的输出，以兼容开集识别流程。
            
            # 1. 通过backbone获取最原始的logits和最后一层的特征图
            # 我们的ResNetBackbone被设计为在return_features=True时返回 (logits, feature_map)
            logits, feature_map = self.backbone(x, return_features=True)
            
            # 2. 从特征图计算用于开集识别的512维特征向量 (即分类器前的特征)
            # 注意: avgpool 是 backbone 的一部分
            pooled_features = self.backbone.avgpool(feature_map).view(x.size(0), -1)
            
            # 3. 以字典形式返回，方便后续处理，接口清晰
            return {
                'logits': logits,       # 用于标准的分类任务评估
                'features': pooled_features # 用于原型计算和扩散模型
            }

class SelfSupervisedLosses:
    """自监督学习损失函数集合 (精简版)"""
    @staticmethod
    def contrastive_loss(proj_orig, proj_shuffled, temperature=0.07):
        """对比学习损失 - InfoNCE"""
        proj_orig = F.normalize(proj_orig, dim=1)
        proj_shuffled = F.normalize(proj_shuffled, dim=1)
        logits = torch.mm(proj_orig, proj_shuffled.t()) / temperature
        labels = torch.arange(proj_orig.size(0), device=proj_orig.device)
        return F.cross_entropy(logits, labels)
    
    @staticmethod
    def supervised_contrastive_loss(proj_orig, proj_shuffled, labels, temperature=0.07):
        """
        监督对比学习损失函数 (SupCon Loss)
        它将同一批次内所有相同类别的样本都视为正样本。
        """
        device = proj_orig.device
        batch_size = proj_orig.shape[0]

        # 1. 将原始投影和打乱投影合并，视为一个样本的两种'视图'
        features = torch.cat([proj_orig, proj_shuffled], dim=0)
        # 对应的标签也要复制一份
        labels = torch.cat([labels, labels], dim=0)
        
        # 2. 归一化特征
        features = F.normalize(features, dim=1)
        
        # 3. 计算所有样本对之间的相似度矩阵
        # 维度: [2*B, 2*B]
        similarity_matrix = torch.matmul(features, features.T) / temperature

        # 4. 创建一个mask来识别正样本对
        #    正样本对：标签相同，但不是同一样本
        
        # 将标签向量扩展成一个二维矩阵，用于逐元素比较
        # labels.view(-1, 1) -> 列向量, labels.view(1, -1) -> 行向量
        # mask[i, j] is True if labels[i] == labels[j]
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)

        # 排除对角线上的自身与自身的比较
        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size, device=device)
        mask = mask * logits_mask
        
        # 5. 计算损失
        # SupCon Loss 的标准计算方式
        # 对于每个样本，分母是它与其他所有样本的相似度指数和
        # 分子是它与所有正样本的相似度指数和
        
        # 为了数值稳定性，我们先减去每行的最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # 计算分母 (log-sum-exp)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算分子 (只关心正样本对的log-prob)
        # mask.sum(1) 是每个样本的正样本数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 最终损失是所有样本损失的均值的相反数
        loss = -mean_log_prob_pos.mean()
        
        return loss
        
    @staticmethod
    def consistency_loss(features_orig, features_shuffled):
        """特征一致性损失"""
        orig_flat = F.adaptive_avg_pool2d(features_orig, (1, 1)).view(features_orig.size(0), -1)
        shuffled_flat = F.adaptive_avg_pool2d(features_shuffled, (1, 1)).view(features_shuffled.size(0), -1)
        orig_flat = F.normalize(orig_flat, dim=1)
        shuffled_flat = F.normalize(shuffled_flat, dim=1)
        cosine_sim = torch.sum(orig_flat * shuffled_flat, dim=1)
        return 1 - cosine_sim.mean()

def compute_enhanced_adversarial_loss(outputs, targets):
    """计算增强的对抗损失 (精简版)"""
    batch_size = targets.size(0)
    
    # 1. 分类损失
    cls_loss_orig = F.cross_entropy(outputs['logits_orig'], targets)
    cls_loss_shuffled = sum(F.cross_entropy(logit_s, targets) for logit_s in outputs['logits_shuffled'])
    cls_loss = cls_loss_orig + cls_loss_shuffled
    
    # 2. 自监督学习损失 (只保留对比和一致性)
    ssl_losses = SelfSupervisedLosses()
    total_ssl_loss = torch.tensor(0.0, device=targets.device)
    for i in range(len(outputs['logits_shuffled'])):
        # ==================== 修改下面这一行 ====================
        # 使用新的监督对比损失，并传入 'targets' (也就是 labels)
        total_ssl_loss += 0.1 * ssl_losses.supervised_contrastive_loss(
            outputs['proj_orig'], 
            outputs['proj_shuffled'][i], 
            targets,  # <--- 这是关键的新增参数
            temperature=0.07
        )
        # ==================== 修改结束 ====================
        total_ssl_loss += 0.05 * ssl_losses.consistency_loss(outputs['features_orig'], outputs['features_shuffled'][i])

    # 3. 对抗学习损失
    real_labels = torch.zeros(batch_size, dtype=torch.long, device=targets.device)
    fake_labels = torch.ones(batch_size, dtype=torch.long, device=targets.device)
    
    disc_loss_real = F.cross_entropy(outputs['disc_orig'], real_labels)
    disc_loss_fake = sum(F.cross_entropy(disc_s, fake_labels) for disc_s in outputs['disc_shuffled'])
    disc_loss = disc_loss_real + disc_loss_fake
    
    # === 移除了 domain_loss 的计算 ===
    
    # 4. 噪声检测损失
    noise_consistency_loss = -F.mse_loss(outputs['noise_map_orig'], torch.zeros_like(outputs['noise_map_orig']))
    noise_consistency_loss += sum(F.mse_loss(noise_map_s, torch.ones_like(noise_map_s) * 0.8) for noise_map_s in outputs['noise_map_shuffled'])
    
    # 组合总损失 (移除了 domain_loss)
    total_adv_loss = (0.1 * disc_loss + 0.05 * noise_consistency_loss)
    total_loss = cls_loss + total_ssl_loss + total_adv_loss
    
    return {
        'cls_loss': cls_loss,
        'ssl_loss': total_ssl_loss,
        'adv_loss': total_adv_loss,
        'total_loss': total_loss
    }

# 1. Sinusoidal time embedding for diffusion model
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# 2. The Diffusion Model for Feature Denoising
class FeatureDiffusionModel(nn.Module):
    def __init__(self, feature_dim=512, num_classes=150, time_dim=256, hidden_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        self.class_emb = nn.Embedding(num_classes, time_dim)
        
        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim + 2 * time_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, noisy_features, t, class_labels):
        time_emb = self.time_mlp(t)
        class_emb = self.class_emb(class_labels)
        x = torch.cat([noisy_features, time_emb, class_emb], dim=-1)
        return self.denoise_net(x)

# 3. The main wrapper model for the entire Open-Set pipeline
class OrchidOpenSetModel(nn.Module):
    def __init__(self, num_known_classes=150, feature_dim=512):
        super().__init__()
        # 初始化您原来的模型作为特征提取器
        self.feature_extractor = DCLNTSHybridNet(num_classes=num_known_classes)
        # 初始化扩散模型
        self.diffusion_model = FeatureDiffusionModel(feature_dim=feature_dim, num_classes=num_known_classes)

    def freeze_feature_extractor(self, freeze=True):
        print(f"Feature extractor frozen: {freeze}")
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
            
    def get_features(self, x):
        """A dedicated method to get features during evaluation."""
        self.feature_extractor.eval()
        with torch.no_grad():
            _, feature_map = self.feature_extractor.backbone(x, return_features=True)
            pooled_features = self.feature_extractor.backbone.avgpool(feature_map).view(x.size(0), -1)
        return pooled_features