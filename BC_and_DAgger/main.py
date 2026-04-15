#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Student 2: Behavioral Cloning (BC) and DAgger Implementation
适配特征向量输入版本（数据已经是预处理过的特征）

数据格式：
- images: (500, 75, 1024) - 500个演示，每个75步，1024维特征向量
- actions: (500, 75, 16) - 500个演示，每个75步，16维动作
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import matplotlib.pyplot as plt
import random

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(42)

# ==================== 配置参数 ====================
class Config:
    data_dir = r"D:\0_FILES\6019_Embodied_AI_and_Application\TW\preprocessed"
    batch_size = 64
    learning_rate = 1e-3
    bc_epochs = 100
    dagger_rounds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
print(f"Using device: {cfg.device}")
print(f"Data directory: {cfg.data_dir}")

# ==================== 1. 加载数据 ====================
print("\n" + "=" * 60)
print("Step 1: Loading Data")
print("=" * 60)

# 加载数据
actions = np.load(os.path.join(cfg.data_dir, "liftpot_actions.npy"))
images = np.load(os.path.join(cfg.data_dir, "liftpot_images.npy"))

with open(os.path.join(cfg.data_dir, "stats.json"), "r") as f:
    stats = json.load(f)

action_min = np.array(stats["action_min"])
action_max = np.array(stats["action_max"])
action_dim = len(action_min)

print(f"Images shape: {images.shape}")    # (500, 75, 1024)
print(f"Actions shape: {actions.shape}")  # (500, 75, 16)
print(f"Action dimension: {action_dim}")

# ==================== 2. 数据预处理 ====================
print("\n" + "=" * 60)
print("Step 2: Data Preprocessing")
print("=" * 60)

def normalize_actions(actions, min_val, max_val):
    """将动作归一化到 [0, 1] 范围"""
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (actions - min_val) / range_val

def denormalize_actions(normalized_actions, min_val, max_val):
    """将归一化的动作还原回原始范围"""
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return normalized_actions * range_val + min_val

# 将数据展平为 (总样本数, 特征维度) 和 (总样本数, 动作维度)
num_demos, horizon, feat_dim = images.shape
_, _, act_dim = actions.shape

images_flat = images.reshape(-1, feat_dim)  # (500*75, 1024)
actions_flat = actions.reshape(-1, act_dim)  # (500*75, 16)

print(f"Flattened images shape: {images_flat.shape}")
print(f"Flattened actions shape: {actions_flat.shape}")

# 特征归一化（重要！）
print(f"\nFeature stats before normalization:")
print(f"  Mean: {images_flat.mean():.3f}, Std: {images_flat.std():.3f}")
print(f"  Min: {images_flat.min():.3f}, Max: {images_flat.max():.3f}")

# 归一化特征到 [0, 1] 或标准化
images_normalized = (images_flat - images_flat.min()) / (images_flat.max() - images_flat.min() + 1e-8)
print(f"\nAfter normalization:")
print(f"  Min: {images_normalized.min():.3f}, Max: {images_normalized.max():.3f}")

# 动作归一化
actions_normalized = normalize_actions(actions_flat, action_min, action_max)
print(f"Normalized actions - min: {actions_normalized.min():.3f}, max: {actions_normalized.max():.3f}")

# 划分训练集和验证集
num_samples = len(images_normalized)
indices = np.random.permutation(num_samples)
split = int(0.8 * num_samples)
train_idx = indices[:split]
val_idx = indices[split:]

X_train = images_normalized[train_idx]
y_train = actions_normalized[train_idx]
X_val = images_normalized[val_idx]
y_val = actions_normalized[val_idx]

print(f"\nTrain samples: {len(X_train)}, Val samples: {len(X_val)}")

# ==================== 3. 定义神经网络模型（MLP，因为输入是特征向量）====================
print("\n" + "=" * 60)
print("Step 3: Defining Neural Network")
print("=" * 60)

class BehavioralCloningNet(nn.Module):
    """MLP 网络，输入特征向量，输出动作"""
    def __init__(self, input_dim=1024, action_dim=16, hidden_dims=[512, 256, 128]):
        super(BehavioralCloningNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Sigmoid())  # 输出范围 [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 获取输入维度
input_dim = X_train.shape[1]
print(f"Input dimension: {input_dim}")

model = BehavioralCloningNet(
    input_dim=input_dim, 
    action_dim=action_dim,
    hidden_dims=[512, 256, 128]
).to(cfg.device)

print(f"Model architecture:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# ==================== 4. 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, actions in loader:
        features = features.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, actions in loader:
            features = features.to(device)
            actions = actions.to(device)
            outputs = model(features)
            loss = criterion(outputs, actions)
            total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)

def train_model(model, X_train, y_train, X_val, y_val, epochs, device):
    """完整的训练流程"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.data_dir, "bc_model_best.pth"))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return train_losses, val_losses

# ==================== 5. 训练 BC 模型 ====================
print("\n" + "=" * 60)
print("Step 4: Training BC Model")
print("=" * 60)

train_losses, val_losses = train_model(
    model, X_train, y_train, X_val, y_val, 
    epochs=cfg.bc_epochs, 
    device=cfg.device
)

print("\nTraining completed!")

# 保存最终模型
model_path = os.path.join(cfg.data_dir, "bc_model_final.pth")
torch.save(model.state_dict(), model_path)
print(f"Final model saved to {model_path}")

# 绘制训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('BC Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(train_losses, label='Train Loss')
plt.semilogy(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Log MSE Loss')
plt.title('BC Training Curve (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.data_dir, "bc_training_curve.png"))
plt.show()

# ==================== 6. 评估模型 ====================
print("\n" + "=" * 60)
print("Step 5: Evaluating BC Model")
print("=" * 60)

def evaluate_model(model, X_val, y_val, action_min, action_max, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        pred_normalized = model(X_tensor).cpu().numpy()
    
    # 反归一化
    pred_actions = denormalize_actions(pred_normalized, action_min, action_max)
    true_actions = denormalize_actions(y_val, action_min, action_max)
    
    # 计算误差
    mse = np.mean((pred_actions - true_actions) ** 2)
    mae = np.mean(np.abs(pred_actions - true_actions))
    mae_per_dim = np.mean(np.abs(pred_actions - true_actions), axis=0)
    
    print(f"Overall MSE: {mse:.6f}")
    print(f"Overall MAE: {mae:.6f}")
    print(f"MAE per dimension - Min: {mae_per_dim.min():.6f}, Max: {mae_per_dim.max():.6f}, Mean: {mae_per_dim.mean():.6f}")
    
    return pred_actions, true_actions, mae_per_dim

pred_actions, true_actions, mae_per_dim = evaluate_model(
    model, X_val, y_val, action_min, action_max, cfg.device
)

# 绘制每个维度的MAE
plt.figure(figsize=(14, 5))
plt.bar(range(action_dim), mae_per_dim)
plt.xlabel('Action Dimension')
plt.ylabel('MAE')
plt.title('MAE per Action Dimension')
plt.xticks(range(action_dim), rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.data_dir, "bc_mae_per_dim.png"))
plt.show()

# 绘制预测 vs 真实值散点图（采样部分维度）
plt.figure(figsize=(14, 8))
sample_size = min(500, len(pred_actions))
sample_indices = np.random.choice(len(pred_actions), sample_size, replace=False)

for i in range(min(6, action_dim)):
    plt.subplot(2, 3, i+1)
    plt.scatter(true_actions[sample_indices, i], pred_actions[sample_indices, i], alpha=0.5, s=1)
    plt.plot([true_actions[:, i].min(), true_actions[:, i].max()], 
             [true_actions[:, i].min(), true_actions[:, i].max()], 
             'r--', label='Perfect prediction')
    plt.xlabel('True Action')
    plt.ylabel('Predicted Action')
    plt.title(f'Action Dim {i}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.data_dir, "bc_prediction_scatter.png"))
plt.show()

# ==================== 7. Covariate Shift 分析 ====================
print("\n" + "=" * 60)
print("Step 6: Covariate Shift Analysis")
print("=" * 60)

# 分析不同时间步的误差累积
def analyze_temporal_error(images, actions, model, action_min, action_max, horizon=75):
    """分析随时间步的误差累积"""
    model.eval()
    
    temporal_errors = []
    
    for t in range(horizon):
        # 获取第t步的所有样本
        t_indices = range(t, len(images), horizon)
        t_features = images[t_indices]
        t_actions = actions[t_indices]
        
        with torch.no_grad():
            features_tensor = torch.tensor(t_features, dtype=torch.float32).to(cfg.device)
            pred_normalized = model(features_tensor).cpu().numpy()
        
        pred_actions = denormalize_actions(pred_normalized, action_min, action_max)
        true_actions = denormalize_actions(t_actions, action_min, action_max)
        
        mae = np.mean(np.abs(pred_actions - true_actions))
        temporal_errors.append(mae)
    
    return temporal_errors

# 计算时间步误差
temporal_errors = analyze_temporal_error(
    images_normalized, actions_normalized, model, action_min, action_max
)

plt.figure(figsize=(10, 5))
plt.plot(temporal_errors, marker='o')
plt.xlabel('Time Step')
plt.ylabel('MAE')
plt.title('Error Accumulation Over Time (Covariate Shift Analysis)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(cfg.data_dir, "covariate_shift_analysis.png"))
plt.show()

print(f"Error at step 1: {temporal_errors[0]:.6f}")
print(f"Error at step {len(temporal_errors)-1}: {temporal_errors[-1]:.6f}")
print(f"Error growth: {(temporal_errors[-1] / temporal_errors[0] - 1) * 100:.1f}%")

# ==================== 8. DAgger 简化实现 ====================
print("\n" + "=" * 60)
print("Step 7: DAgger Implementation")
print("=" * 60)

def run_dagger(model, X_train, y_train, X_val, y_val, action_min, action_max, device, num_rounds=5):
    """DAgger 训练"""
    dagger_X = list(X_train)
    dagger_y = list(y_train)
    
    round_results = []
    
    for round_idx in range(num_rounds):
        print(f"\n--- DAgger Round {round_idx + 1}/{num_rounds} ---")
        
        # 生成新数据（使用策略的轨迹）
        new_X = []
        new_y = []
        
        # 模拟策略 rollout
        num_trajectories = 20
        traj_length = 50
        
        for traj_idx in range(num_trajectories):
            # 随机选择起始点
            start_idx = np.random.randint(0, len(X_val) - traj_length)
            current_features = X_val[start_idx]
            
            for step in range(traj_length):
                # 模型预测动作
                model.eval()
                with torch.no_grad():
                    feat_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_normalized = model(feat_tensor).cpu().numpy()[0]
                
                # 获取专家动作（从验证集中取）
                expert_action = y_val[min(start_idx + step, len(y_val)-1)]
                
                # DAgger 混合策略
                beta = max(0.1, 1.0 - (round_idx + 1) / num_rounds)
                if np.random.random() < beta:
                    final_action = expert_action
                else:
                    final_action = pred_normalized
                
                new_X.append(current_features)
                new_y.append(expert_action)
                
                # 更新状态（简化的状态转移）
                if step + 1 < len(X_val):
                    current_features = X_val[min(start_idx + step + 1, len(X_val)-1)]
        
        # 添加到数据集
        dagger_X.extend(new_X)
        dagger_y.extend(new_y)
        
        print(f"Added {len(new_X)} new samples")
        print(f"Total dataset size: {len(dagger_X)}")
        
        # 重新训练
        dagger_X_np = np.array(dagger_X)
        dagger_y_np = np.array(dagger_y)
        
        # 划分新的训练/验证集
        split_idx = int(0.8 * len(dagger_X_np))
        new_train_X = dagger_X_np[:split_idx]
        new_train_y = dagger_y_np[:split_idx]
        new_val_X = dagger_X_np[split_idx:]
        new_val_y = dagger_y_np[split_idx:]
        
        # 训练
        train_losses, val_losses = train_model(
            model, new_train_X, new_train_y, new_val_X, new_val_y,
            epochs=20, device=device
        )
        
        round_results.append({
            'round': round_idx + 1,
            'dataset_size': len(dagger_X),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None
        })
        
        print(f"Round {round_idx + 1} completed. Val Loss: {round_results[-1]['final_val_loss']:.6f}")
    
    return round_results

# 运行 DAgger
dagger_results = run_dagger(
    model, X_train, y_train, X_val, y_val,
    action_min, action_max, cfg.device,
    num_rounds=cfg.dagger_rounds
)

# 绘制 DAgger 进度
if dagger_results:
    plt.figure(figsize=(10, 5))
    rounds = [r['round'] for r in dagger_results]
    val_losses = [r['final_val_loss'] for r in dagger_results if r['final_val_loss'] is not None]
    plt.plot(rounds[:len(val_losses)], val_losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('DAgger Round')
    plt.ylabel('Validation Loss')
    plt.title('DAgger Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(cfg.data_dir, "dagger_progress.png"))
    plt.show()

# ==================== 9. 保存结果 ====================
print("\n" + "=" * 60)
print("Step 8: Saving Results")
print("=" * 60)

# 保存结果
results = {
    "bc_train_losses": train_losses,
    "bc_val_losses": val_losses,
    "final_train_loss": train_losses[-1],
    "final_val_loss": val_losses[-1],
    "mae_per_dim": mae_per_dim.tolist(),
    "temporal_errors": temporal_errors,
    "dagger_results": dagger_results,
    "action_dim": action_dim,
    "input_dim": input_dim,
}

with open(os.path.join(cfg.data_dir, "bc_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {os.path.join(cfg.data_dir, 'bc_results.json')}")

print("\n" + "=" * 60)
print("All Tasks Completed Successfully!")
print("=" * 60)
print("\nDeliverables generated:")
print("1. bc_model_final.pth - Final BC model")
print("2. bc_model_best.pth - Best BC model")
print("3. bc_training_curve.png - Training curve plot")
print("4. bc_mae_per_dim.png - MAE per action dimension")
print("5. bc_prediction_scatter.png - Prediction vs true scatter plots")
print("6. covariate_shift_analysis.png - Temporal error analysis")
print("7. dagger_progress.png - DAgger training progress (if run)")
print("8. bc_results.json - All results data")


# In[ ]:




