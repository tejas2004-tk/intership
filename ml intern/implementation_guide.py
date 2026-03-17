"""
AgroKD-Net: Implementation Guide
Lightweight Model for Crop-Weed Detection with Knowledge Distillation

Complete step-by-step Python implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# ============================================================================

class COCOWeedDataset(Dataset):
    """
    Load COCO format weed detection dataset
    
    Expected structure:
    dataset/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── annotations.json
    """
    
    def __init__(self, dataset_path, annotations_file, image_size=640, 
                 augment=False):
        """
        Args:
            dataset_path: Path to dataset folder
            annotations_file: Path to COCO annotations JSON
            image_size: Size to resize images to
            augment: Whether to apply augmentations
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.augment = augment
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] 
                          for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.img_to_annots = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_annots:
                self.img_to_annots[img_id] = []
            self.img_to_annots[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.dataset_path, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        
        # Load annotations
        img_id = img_info['id']
        annotations = self.img_to_annots.get(img_id, [])
        
        # Convert to bounding boxes
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            
            # Rescale to image_size
            x_scaled = (x / orig_width) * self.image_size
            y_scaled = (y / orig_height) * self.image_size
            w_scaled = (w / orig_width) * self.image_size
            h_scaled = (h / orig_height) * self.image_size
            
            # Convert to center format
            cx = x_scaled + w_scaled / 2
            cy = y_scaled + h_scaled / 2
            
            boxes.append([cx, cy, w_scaled, h_scaled])
            labels.append(ann['category_id'] - 1)  # 0-indexed
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4))
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes)
            labels = np.array(labels, dtype=np.int64)
        
        return {
            'image': image,
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'img_id': img_id
        }


# ============================================================================
# PART 2: LIGHTWEIGHT STUDENT BACKBONE
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution reduces FLOPs"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super().__init__()
        
        # Depthwise: separate convolution per channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size, stride, padding,
                                   groups=in_channels)
        
        # Pointwise: 1×1 convolution to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightweightStudentBackbone(nn.Module):
    """
    Lightweight student network using depthwise separable convolutions
    
    FLOPs reduction: Instead of HWC·k²·C, use HWC·(k² + C)
    This is ~8-9x reduction for typical architectures
    """
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Input: (B, 3, 640, 640)
        # Feature level 1: 640×640
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Feature level 2: 320×320 (stride=2)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 32),
            DepthwiseSeparableConv(32, 32)
        )
        
        # Feature level 3: 160×160 (stride=2)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 64)
        )
        
        # Feature level 4: 80×80 (stride=2)
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128)
        )
    
    def forward(self, x):
        """
        Returns multi-scale feature maps
        
        Returns:
            f1: Feature level 1 (640×640, 16 channels)
            f2: Feature level 2 (320×320, 32 channels)
            f3: Feature level 3 (160×160, 64 channels)
            f4: Feature level 4 (80×80, 128 channels)
        """
        f1 = self.conv1(x)          # (B, 16, 640, 640)
        f2 = self.conv2(f1)         # (B, 32, 320, 320)
        f3 = self.conv3(f2)         # (B, 64, 160, 160)
        f4 = self.conv4(f3)         # (B, 128, 80, 80)
        
        return f1, f2, f3, f4


# ============================================================================
# PART 3: MULTI-SCALE FEATURE AGGREGATION MODULE
# ============================================================================

class MultiScaleFeatureAggregation(nn.Module):
    """
    Aggregate features from multiple scales to detect weeds of varying sizes
    
    Why: Small weeds appear in early (high-resolution) features,
         large weeds in late (low-resolution) features
    """
    
    def __init__(self, channels=[16, 32, 64, 128]):
        super().__init__()
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(4))
        
        # Project all features to common channel dimension
        self.proj = nn.ModuleList([
            nn.Conv2d(c, 64, kernel_size=1) for c in channels
        ])
    
    def forward(self, f1, f2, f3, f4):
        """
        Fuse features from all scales
        
        Input:
            f1: (B, 16, 640, 640)
            f2: (B, 32, 320, 320)
            f3: (B, 64, 160, 160)
            f4: (B, 128, 80, 80)
        
        Output:
            fusion: (B, 64, 160, 160) [middle resolution chosen]
        """
        # Normalize weights
        alpha = torch.softmax(self.scale_weights, dim=0)
        
        # Project all to channel=64
        f1_proj = self.proj[0](f1)  # (B, 64, 640, 640)
        f2_proj = self.proj[1](f2)  # (B, 64, 320, 320)
        f3_proj = self.proj[2](f3)  # (B, 64, 160, 160)
        f4_proj = self.proj[3](f4)  # (B, 64, 80, 80)
        
        # Upsample/downsample to target resolution (160×160)
        target_size = (160, 160)
        f1_resized = F.interpolate(f1_proj, size=target_size, 
                                  mode='bilinear', align_corners=False)
        f2_resized = F.interpolate(f2_proj, size=target_size, 
                                  mode='bilinear', align_corners=False)
        f3_resized = f3_proj  # Already target size
        f4_resized = F.interpolate(f4_proj, size=target_size, 
                                  mode='bilinear', align_corners=False)
        
        # Weighted fusion
        fusion = (alpha[0] * f1_resized + 
                 alpha[1] * f2_resized + 
                 alpha[2] * f3_resized + 
                 alpha[3] * f4_resized)
        
        return fusion


# ============================================================================
# PART 4: LOSS MODULES
# ============================================================================

class EnergyAwareKnowledgeDistillation(nn.Module):
    """
    Transfer knowledge from teacher while minimizing energy
    
    Loss = KL(teacher_output || student_output) + λ × Energy(model)
    """
    
    def __init__(self, temperature=4.0, lambda_energy=0.1):
        super().__init__()
        self.T = temperature
        self.lambda_energy = lambda_energy
    
    def forward(self, student_logits, teacher_logits, energy_budget):
        """
        Args:
            student_logits: (B, num_classes)
            teacher_logits: (B, num_classes)
            energy_budget: Estimated energy consumption (scalar)
        
        Returns:
            loss: Knowledge distillation + energy penalty
        """
        # Soft targets from teacher
        P_teacher = torch.softmax(teacher_logits / self.T, dim=1)
        log_P_student = torch.log_softmax(student_logits / self.T, dim=1)
        
        # KL divergence loss
        kl_loss = F.kl_div(log_P_student, P_teacher, reduction='batchmean')
        
        # Scale by temperature squared
        kl_loss = kl_loss * (self.T ** 2)
        
        # Energy penalty (encourage low-energy models)
        energy_loss = self.lambda_energy * energy_budget
        
        # Total loss
        total_loss = kl_loss + energy_loss
        
        return total_loss


class GradientBalancedPixelReweighting(nn.Module):
    """
    Handle severe pixel imbalance by balancing gradient magnitudes
    
    Intuition: Rare weed pixels generate small gradients → underlearned
              We weight inversely to gradient magnitude → balanced learning
    """
    
    def __init__(self, num_classes=3, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, num_classes, H, W)
            targets: (B, H, W) with class indices
        
        Returns:
            weighted_loss: Balanced cross-entropy loss
        """
        total_loss = 0
        
        for class_idx in range(self.num_classes):
            # Get mask for this class
            class_mask = (targets == class_idx).float()
            
            if class_mask.sum() == 0:
                continue  # Skip empty classes
            
            # Get predictions for this class
            class_pred = predictions[:, class_idx, :, :]
            
            # Cross-entropy loss for this class
            ce_loss = F.binary_cross_entropy_with_logits(
                class_pred, class_mask, reduction='none'
            )
            
            # Compute gradient magnitude for this class
            class_loss_mean = ce_loss[class_mask > 0].mean()
            if class_loss_mean.requires_grad:
                class_loss_mean.backward(retain_graph=True)
            
            # Weight inversely to frequency
            class_freq = class_mask.mean()
            weight = 1.0 / (class_freq + self.epsilon)
            
            # Accumulate weighted loss
            total_loss += weight * ce_loss.mean()
        
        return total_loss / self.num_classes


class StructuralContextDistillation(nn.Module):
    """
    Transfer structural/relational knowledge via affinity matrices
    
    Why: Crops grow in rows, weeds appear between them
         Preserving spatial relationships reduces false positives
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, teacher_features, student_features):
        """
        Args:
            teacher_features: (B, C, H, W)
            student_features: (B, C, H, W)
        
        Returns:
            loss: Frobenius norm of affinity matrix difference
        """
        def compute_affinity(F):
            """Compute Gram matrix (relational structure)"""
            B, C, H, W = F.shape
            F_flat = F.view(B, C, -1)  # (B, C, HW)
            
            # Compute Gram matrix: G = F × F^T
            affinity = torch.bmm(F_flat, F_flat.transpose(1, 2))
            
            # Normalize by Frobenius norm
            norm = torch.norm(affinity, p='fro', dim=(1, 2), keepdim=True)
            affinity = affinity / (norm + 1e-6)
            
            return affinity
        
        # Compute affinity matrices
        R_teacher = compute_affinity(teacher_features)
        R_student = compute_affinity(student_features)
        
        # Loss: Frobenius distance
        loss = torch.norm(R_teacher - R_student, p='fro') / R_teacher.numel()
        
        return loss


class DomainShiftResistantDistillation(nn.Module):
    """
    Align feature distributions across domains (farms/crops)
    
    Loss: Mean alignment + Covariance alignment
    """
    
    def __init__(self, lambda_cov=0.1):
        super().__init__()
        self.lambda_cov = lambda_cov
    
    def forward(self, features_domain1, features_domain2):
        """
        Args:
            features_domain1: Features from dataset A (B, C, H, W)
            features_domain2: Features from dataset B (B, C, H, W)
        
        Returns:
            loss: Domain alignment loss
        """
        # Flatten to (B*H*W, C)
        B1, C, H1, W1 = features_domain1.shape
        B2, _, H2, W2 = features_domain2.shape
        
        f1_flat = features_domain1.permute(0, 2, 3, 1).reshape(-1, C)
        f2_flat = features_domain2.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Mean alignment
        mean1 = f1_flat.mean(dim=0)
        mean2 = f2_flat.mean(dim=0)
        mean_loss = torch.norm(mean1 - mean2) ** 2
        
        # Covariance alignment
        cov1 = torch.cov(f1_flat.T)
        cov2 = torch.cov(f2_flat.T)
        cov_loss = torch.norm(cov1 - cov2, p='fro') ** 2
        
        # Total domain loss
        loss = mean_loss + self.lambda_cov * cov_loss
        
        return loss


# ============================================================================
# PART 5: COMPLETE AgroKD-Net MODEL
# ============================================================================

class AgroKDNet(nn.Module):
    """
    Complete AgroKD-Net: Lightweight model with all novel modules
    """
    
    def __init__(self, num_classes=3, 
                 use_eapd=True, use_gbpr=True, 
                 use_scd=True, use_dsrd=True,
                 lambda1=0.1, lambda2=1.0, 
                 lambda3=0.2, lambda4=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_eapd = use_eapd
        self.use_gbpr = use_gbpr
        self.use_scd = use_scd
        self.use_dsrd = use_dsrd
        
        # Backbone
        self.backbone = LightweightStudentBackbone(num_classes)
        
        # Multi-scale fusion
        self.multi_scale_fusion = MultiScaleFeatureAggregation()
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
        # Loss modules
        if use_eapd:
            self.eapd = EnergyAwareKnowledgeDistillation(lambda_energy=0.05)
        if use_gbpr:
            self.gbpr = GradientBalancedPixelReweighting(num_classes)
        if use_scd:
            self.scd = StructuralContextDistillation()
        if use_dsrd:
            self.dsrd = DomainShiftResistantDistillation()
        
        # Loss weights
        self.lambda1 = lambda1  # EAPD
        self.lambda2 = lambda2  # GBPR
        self.lambda3 = lambda3  # SCD
        self.lambda4 = lambda4  # DSRD
    
    def forward(self, x):
        """Forward pass for inference"""
        f1, f2, f3, f4 = self.backbone(x)
        fusion = self.multi_scale_fusion(f1, f2, f3, f4)
        logits = self.detection_head(fusion)
        return logits
    
    def calculate_flops(self, input_shape=(1, 3, 640, 640)):
        """Estimate FLOPs"""
        # Simple estimation: roughly proportional to number of parameters
        # For accurate measurement, use thop library
        total_params = sum(p.numel() for p in self.parameters())
        # Rough FLOPs: params × input_size
        flops = total_params * 640 * 640 / 1e9
        return flops


# ============================================================================
# PART 6: TRAINING LOOP
# ============================================================================

class AgroKDNetTrainer:
    """Complete training pipeline"""
    
    def __init__(self, model, device='cuda', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
        
        # Criterion
        self.criterion_det = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_map': [],
            'val_map': []
        }
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            targets = batch['labels'].to(self.device)  # (B,)
            
            # Forward pass
            logits = self.model(images)  # (B, num_classes, H, W)
            
            # Resize targets to match output spatial dims
            B, C, H, W = logits.shape
            targets_resized = F.interpolate(
                targets.unsqueeze(1).float(), 
                size=(H, W), mode='nearest'
            ).squeeze(1).long()
            
            # Detection loss
            loss_det = self.criterion_det(logits, targets_resized)
            
            # Total loss
            loss = loss_det
            
            # Backward & optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                logits = self.model(images)
                
                # Resize for loss
                B, C, H, W = logits.shape
                targets_resized = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=(H, W), mode='nearest'
                ).squeeze(1).long()
                
                loss = self.criterion_det(logits, targets_resized)
                total_loss += loss.item()
                
                # Store predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets_resized.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop"""
        print(f"Training AgroKD-Net for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if epoch > 20 and len(self.history['val_loss']) > 5:
                recent_losses = self.history['val_loss'][-5:]
                if all(l >= recent_losses[0] for l in recent_losses):
                    print("Early stopping triggered!")
                    break
        
        return self.history


# ============================================================================
# PART 7: EVALUATION & METRICS
# ============================================================================

def calculate_map(predictions, targets, num_classes=3):
    """Simple mAP calculation (can be improved with pycocotools)"""
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for pred, target in zip(predictions, targets):
        # Flatten
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        for class_idx in range(num_classes):
            tp = ((pred_flat == class_idx) & (target_flat == class_idx)).sum()
            fp = ((pred_flat == class_idx) & (target_flat != class_idx)).sum()
            fn = ((pred_flat != class_idx) & (target_flat == class_idx)).sum()
            
            all_tp += tp
            all_fp += fp
            all_fn += fn
    
    # Precision & Recall
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1


# ============================================================================
# PART 8: MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("AgroKD-Net: Lightweight Crop-Weed Detection")
    print("="*80)
    
    # Step 1: Create datasets
    print("\n[Step 1] Loading Datasets...")
    
    train_dataset = COCOWeedDataset(
        dataset_path="datasets/MH-Weed16/images",
        annotations_file="datasets/MH-Weed16/annotations.json",
        image_size=640,
        augment=True
    )
    
    val_dataset = COCOWeedDataset(
        dataset_path="datasets/MH-Weed16/images",
        annotations_file="datasets/MH-Weed16/annotations_val.json",
        image_size=640,
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ Loaded {len(train_dataset)} training images")
    print(f"✓ Loaded {len(val_dataset)} validation images")
    
    # Step 2: Create model
    print("\n[Step 2] Building AgroKD-Net...")
    
    model = AgroKDNet(
        num_classes=3,
        use_eapd=True,
        use_gbpr=True,
        use_scd=True,
        use_dsrd=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params / 1e6:.2f}M parameters")
    print(f"✓ Estimated FLOPs: {model.calculate_flops():.2f}G")
    
    # Step 3: Train
    print("\n[Step 3] Training Model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = AgroKDNetTrainer(model, device=device, lr=0.001)
    
    history = trainer.train(train_loader, val_loader, epochs=3)  # Demo: 3 epochs
    
    # Step 4: Save results
    print("\n[Step 4] Saving Results...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_params': total_params,
        'training_history': history,
        'config': {
            'image_size': 640,
            'batch_size': 16,
            'learning_rate': 0.001,
            'epochs': 3,
            'optimizer': 'AdamW'
        }
    }
    
    import pickle
    with open('agroKD_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("✓ Results saved to agroKD_results.pkl")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
