import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================== Load Configuration ==================
# 修改: 读取消融实验的配置文件
with open('ablation_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model configuration
selected_model = config['selected_model']
model_config = config['models'][selected_model]
training_config = config['trainings'][selected_model]
data_config = config['datas'][selected_model]
logging_config = config['loggings'][selected_model]

# Path configuration
backbone = logging_config['backbone']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']
os.makedirs(result_dir, exist_ok=True)

# ================== Set Random Seed ==================
def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(training_config['seed'])

# ================== Data Loading ==================
print("\n========== Loading Data ==========")
data_path = data_config['data_path']
data = torch.load(data_path, map_location='cpu')
ns_all = data['vorticity']  # Original shape: [1280, 100, 128, 128]
print(f"Original data shape: {ns_all.shape}")

# Dataset split (与原始推理代码保持一致)
total_samples = ns_all.shape[0]
train_end = int(0.8 * total_samples)
val_end = int(0.9 * total_samples)
test_data = ns_all[val_end:]              # 使用训练脚本中定义的测试集部分
test_data = test_data.unsqueeze(2)        # Add channel dimension -> [128, 100, 1, 128, 128]
print(f"Test data shape: {test_data.shape}")

# ================== Model Initialization ==================
print("\n========== Initializing Model ==========")
# ================================================================
# 修改: 适配消融实验的模型加载逻辑
# 导入统一的模型创建工厂
from model.triton_variants import create_model 

# 使用 partial 构建与训练脚本完全一致的 model_dict
model_dict = {
    'Triton_full': partial(create_model, model_type='full'),
    'Triton_flat': partial(create_model, model_type='flat'),
    'Triton_no_skip': partial(create_model, model_type='no_skip'),
    'Triton_no_ldc': partial(create_model, model_type='no_ldc'),
}
# ================================================================

# Instantiate the model (此部分逻辑与训练脚本保持一致)
if selected_model in model_dict:
    ModelClass = model_dict[selected_model]
    model_params = model_config['parameters']
    model = ModelClass(**model_params).to(device)
    model.eval()
    print(f"'{selected_model}' has been loaded successfully.")
else:
    raise ValueError(f"Model '{selected_model}' is not defined in the model registry.")

# ================== Load Model Weights ==================
print("\n========== Loading Model Weights ==========")
best_model_path = os.path.join(checkpoint_dir, f"{backbone}_best_model.pth")
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # Handle possible DataParallel/DistributedDataParallel wrapping
    # 训练脚本中保存的是 unwrapped model state_dict，但为保险起见保留此逻辑
    new_state_dict = OrderedDict()
    is_wrapped = all(k.startswith('module.') for k in checkpoint.keys())
    if is_wrapped:
        for k, v in checkpoint.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    print(f"Successfully loaded model weights from: {best_model_path}")
else:
    raise FileNotFoundError(f"Model checkpoint not found at: {best_model_path}")

# ================== Inference Configuration ==================
torch.set_grad_enabled(False)
# 预测未来 99 个时间步 (t=1 to t=99)
rollout_steps = 99
input_length = data_config['input_length']
variables_input = data_config.get('variables_input', [0])
variables_output = data_config.get('variables_output', [0])
downsample_factor = data_config['downsample_factor']

# Dimension calculation
original_H, original_W = test_data.shape[-2], test_data.shape[-1]
H = original_H // downsample_factor
W = original_W // downsample_factor

# ================== Result Containers ==================
num_samples = test_data.size(0)
all_inputs = np.zeros((num_samples, input_length, len(variables_input), H, W), dtype=np.float32)
all_outputs = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)
all_targets = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)

# ================== Visualization Settings ==================
viz_dir = os.path.join(result_dir, f"{backbone}_visualizations")
os.makedirs(viz_dir, exist_ok=True)

def visualize_comparison(pred, true, step, save_path):
    """Visualization comparison function"""
    plt.figure(figsize=(16, 6), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='RdBu_r', vmin=-3, vmax=3)
    plt.title(f'Prediction @ Step {step+1}', fontsize=12) # step_idx from 0-98 -> Step 1-99
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 2, 2)
    plt.imshow(true, cmap='RdBu_r', vmin=-3, vmax=3)
    plt.title(f'Ground Truth @ Step {step+1}', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ================== Main Inference Loop ==================
print("\n========== Starting Inference ==========")
total_steps = num_samples * rollout_steps
with tqdm(total=total_steps, desc="Total Progress", unit="step", position=0) as pbar_total:
    for sample_idx in tqdm(range(num_samples), desc="Processing Samples", unit="sample", position=1):
        # Current sample data [100, 1, 128, 128]
        current_sample = test_data[sample_idx]
        
        # Create visualization directory for the sample
        sample_viz_dir = os.path.join(viz_dir, f"sample_{sample_idx:04d}")
        os.makedirs(sample_viz_dir, exist_ok=True)
        
        # ===== Initial Condition Handling =====
        initial_input = current_sample[:input_length, variables_input, ::downsample_factor, ::downsample_factor]
        all_inputs[sample_idx] = initial_input.cpu().numpy()
        
        # ===== Ground Truth Handling =====
        ground_truth = current_sample[input_length:input_length+rollout_steps, variables_output, ::downsample_factor, ::downsample_factor]
        all_targets[sample_idx] = ground_truth.cpu().numpy()
        
        # ===== Inference Initialization =====
        inputs = initial_input.clone().to(device) # Shape: [T_in, C, H, W]
        predictions = []
        
        # ===== Autoregressive Time Step Loop =====
        for step_idx in tqdm(range(rollout_steps), desc=f"Sample {sample_idx} Timesteps", leave=False, position=2):
            # Model input requires batch dimension: [B, T_in, C, H, W]
            model_input = inputs.unsqueeze(0).float()
            
            # Model inference
            pred = model(model_input) # Output shape: [B, T_out, C, H, W]
            
            # Extract the prediction, shape: [1, C, H, W]
            # Since T_out=1, pred is [1, 1, C, H, W], we squeeze the time dim.
            last_pred = pred.squeeze(1)
            
            # Save prediction result
            pred_np = last_pred.cpu().numpy() # Shape: [1, C, H, W]
            predictions.append(pred_np[0]) # Append [C, H, W]
            
            # Update the input sequence (rolling window)
            # last_pred shape [1, C, H, W], unsqueeze to [1, 1, C, H, W] to match inputs dim
            inputs = torch.cat([inputs[1:], last_pred], dim=0)
            
            # Save visualization (every 10 steps)
            if step_idx % 10 == 0:
                true_frame = ground_truth[step_idx].squeeze(0).numpy() # [H, W]
                # pred_np[0,0] gets the actual [H,W] array
                viz_path = os.path.join(sample_viz_dir, f"step_{step_idx+1:04d}.png")
                visualize_comparison(pred_np[0, 0], true_frame, step_idx, viz_path)
            
            pbar_total.update(1)
        
        # Save results for the current sample
        # predictions is a list of [C, H, W] arrays, stack them on a new time axis
        all_outputs[sample_idx] = np.stack(predictions, axis=0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================== Save Results ==================
print("\n========== Saving Results ==========")
np.save(os.path.join(result_dir, f"{backbone}_initial_conditions.npy"), all_inputs)
np.save(os.path.join(result_dir, f"{backbone}_predictions.npy"), all_outputs)
np.save(os.path.join(result_dir, f"{backbone}_ground_truth.npy"), all_targets)

# ================== Final Verification ==================
print("\n========== Verifying Results ==========")
print(f"Initial conditions shape: {all_inputs.shape}")
print(f"Predictions shape: {all_outputs.shape}")
print(f"Ground truth shape: {all_targets.shape}")
print(f"Max value check - Initial: {all_inputs.max():.4f}, Predicted: {all_outputs.max():.4f}, True: {all_targets.max():.4f}")
print(f"Min value check - Initial: {all_inputs.min():.4f}, Predicted: {all_outputs.min():.4f}, True: {all_targets.min():.4f}")

print("\n✅ Ablation inference complete! All results have been saved to:", os.path.abspath(result_dir))