# evaluate.py
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from diffusion_utils import denoise_feature
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

def evaluate_open_set(model, known_train_loader, test_loader, device, num_known_classes, alpha=0.5):
    print("\n--- Evaluating Open-Set Performance ---")
    model.eval()
    model.to(device)

    # 1. Compute Prototypes
    print("Step 1: Computing class prototypes...")
    prototypes = torch.zeros(num_known_classes, model.diffusion_model.feature_dim).to(device)
    class_counts = torch.zeros(num_known_classes).to(device)
    
    for images, labels in tqdm(known_train_loader, desc="Calculating Prototypes"):
        images, labels = images.to(device), labels.to(device)
        features = model.get_features(images)
        for i in range(num_known_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                prototypes[i] += features[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    prototypes /= class_counts[:, None]
    print("Prototypes computed.")

    # 2. Get scores for all test samples
    print("Step 2: Getting scores for test set...")
    all_scores, all_labels = [], []
    all_preds_closed_set = []

    for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
        images, labels = images.to(device), labels.to(device)
        features = model.get_features(images)
        
        dist = torch.cdist(features, prototypes)
        min_dist, closest_class_idx = torch.min(dist, dim=1)
        
        recon_error = torch.zeros_like(min_dist)
        for i in range(features.shape[0]):
            z_test = features[i].unsqueeze(0)
            cond_class = closest_class_idx[i].item()
            z_denoised = denoise_feature(model, z_test, cond_class)
            recon_error[i] = F.mse_loss(z_denoised.detach(), z_test)

        scores = alpha * min_dist + (1 - alpha) * recon_error
        
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_preds_closed_set.extend(closest_class_idx.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_preds_closed_set = np.array(all_preds_closed_set)

    # 3. Calculate metrics
    print("\n--- Open-Set Recognition Metrics ---")
    labels_binary = (all_labels >= num_known_classes).astype(int)
    auroc = roc_auc_score(labels_binary, all_scores)
    print(f"AUROC for OSR: {auroc:.4f}")

    # Find optimal threshold to maximize F1 for 'unknown' class
    best_f1 = 0
    best_thresh = 0
    for thresh in np.linspace(all_scores.min(), all_scores.max(), num=1000):
        preds_binary = (all_scores > thresh).astype(int)
        f1 = f1_score(labels_binary, preds_binary, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\nOptimal Threshold found: {best_thresh:.4f} (Maximizes F1-score for 'unknown')")
    
    # Apply threshold for final predictions
    # Known classes are their predicted class, unknown is a new class index (e.g., 150)
    unknown_class_label = num_known_classes 
    preds_open_set = np.where(all_scores > best_thresh, unknown_class_label, all_preds_closed_set)
    
    # Final Metrics
    acc = accuracy_score(all_labels, preds_open_set)
    print(f"Overall Accuracy (Knowns + Unknowns): {acc:.4f}")
    
    # Metrics for Known samples only
    known_mask = all_labels < num_known_classes
    known_acc = accuracy_score(all_labels[known_mask], preds_open_set[known_mask])
    print(f"Accuracy on Known Classes: {known_acc:.4f}")
    
    # Metrics for Unknown samples only
    unknown_mask = all_labels >= num_known_classes
    unknown_recall = recall_score((all_labels[unknown_mask] >= 0), (preds_open_set[unknown_mask] == unknown_class_label)) # All unknown labels are true
    print(f"Recall on Unknown Classes (Ability to detect 'unknowns'): {unknown_recall:.4f}")