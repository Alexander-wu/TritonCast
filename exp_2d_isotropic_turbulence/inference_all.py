import torch
import numpy as np
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 配置加载 ==================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 模型配置
selected_model = config['selected_model']
model_config = config['models'][selected_model]
training_config = config['trainings'][selected_model]
data_config = config['datas'][selected_model]
logging_config = config['loggings'][selected_model]

# 路径配置
backbone = logging_config['backbone']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']
os.makedirs(result_dir, exist_ok=True)

# ================== 随机种子设置 ==================
def set_seed(seed):
    """设置所有随机种子保证可重复性"""
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

# ================== 数据加载 ==================
print("\n========== 加载数据 ==========")
data_path = data_config['data_path']
data = torch.load(data_path, map_location='cpu', weights_only=False)  # 安全加载
ns_all = data['vorticity']  # 原始形状: [1280, 100, 128, 128]

# 数据集划分
total_samples = ns_all.shape[0]
test_data = ns_all[total_samples//10 * 9:]  # 取最后10%作为测试集
test_data = test_data.unsqueeze(2)        # 添加通道维度 [128, 100, 1, 128, 128]

# ================== 模型初始化 ==================
print("\n========== 初始化模型 ==========")
# 模型注册表（必须完整列出所有模型）
from model.triton_model import Triton
from model_baselines.fno import FNO2d 
from model_baselines.dit import Dit
from model_baselines.simvp import SimVP
from model.triton_model_v2 import Triton_v2
from model_baselines.cno import CNO
from model_baselines.mgno import MgNO
from model_baselines.lsm import LSM
from model_baselines.pastnet import PastNetModel
from model_baselines.resnet import ResNet
from model_baselines.unet import U_net

model_dict = {
    'Triton': Triton,
    'Triton_V2': Triton_v2,
    'FNO': FNO2d,
    'DiT': Dit,
    'SimVP': SimVP,
    'CNO': CNO,
    'MGNO': MgNO,
    'LSM': LSM,
    'PastNet': PastNetModel,
    'ResNet': ResNet,
    'U_net': U_net,
}

# 模型实例化
ModelClass = model_dict[selected_model]
model = ModelClass(**model_config['parameters']).to(device)
model.eval()
print(f"{selected_model} has beed loaded")
# ================== 加载模型权重 ==================
print("\n========== 加载模型权重 ==========")
best_model_path = os.path.join(checkpoint_dir, f"{backbone}_best_model.pth")
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # 处理可能的DataParallel包装
    if all(k.startswith('module.') for k in checkpoint.keys()):
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # 去除'module.'前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    print(f"成功加载模型权重：{best_model_path}")
else:
    raise FileNotFoundError(f"模型检查点不存在：{best_model_path}")

# ================== 推理配置 ==================
torch.set_grad_enabled(False)
rollout_steps = 99  # 总预测步数
input_length = data_config['input_length']  # 输入时间步数
variables_input = data_config.get('variables_input', [0])  # 输入变量索引
variables_output = data_config.get('variables_output', [0])  # 输出变量索引
downsample_factor = data_config['downsample_factor']  # 下采样因子

# 维度计算
original_H, original_W = test_data.shape[-2], test_data.shape[-1]  # 原始空间维度
H = original_H // downsample_factor  # 下采样后高度
W = original_W // downsample_factor  # 下采样后宽度

# ================== 结果容器 ==================
num_samples = test_data.size(0)
all_inputs = np.zeros((num_samples, input_length, len(variables_input), H, W), dtype=np.float32)
all_outputs = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)
all_targets = np.zeros((num_samples, rollout_steps, len(variables_output), H, W), dtype=np.float32)

# ================== 可视化设置 ==================
viz_dir = os.path.join(result_dir, f"{backbone}_visualizations")
os.makedirs(viz_dir, exist_ok=True)

def visualize_comparison(pred, true, step, save_path):
    """可视化对比函数"""
    plt.figure(figsize=(16, 6), dpi=150)
    
    plt.subplot(1, 2, 1)
    plt.imshow(pred, cmap='coolwarm', vmin=-3, vmax=3)
    plt.title(f'Prediction @ Step {step}', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 2, 2)
    plt.imshow(true, cmap='coolwarm', vmin=-3, vmax=3)
    plt.title(f'Ground Truth @ Step {step}', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# ================== 主推理循环 ==================
print("\n========== 开始推理 ==========")
total_steps = num_samples * rollout_steps
with tqdm(total=total_steps, desc="总进度", unit="step", position=0) as pbar_total:
    for sample_idx in tqdm(range(num_samples), desc="样本处理", unit="样本", position=1):
        # 当前样本数据 [100, 1, 128, 128]
        current_sample = test_data[sample_idx]
        
        # 创建样本可视化目录
        sample_viz_dir = os.path.join(viz_dir, f"sample_{sample_idx:04d}")
        os.makedirs(sample_viz_dir, exist_ok=True)
        
        # ===== 初始条件处理 =====
        initial_input = current_sample[:input_length, variables_input, ::downsample_factor, ::downsample_factor]
        all_inputs[sample_idx] = initial_input.cpu().numpy()
        
        # ===== 真实标签处理 =====
        ground_truth = current_sample[input_length:input_length+rollout_steps, variables_output, ::downsample_factor, ::downsample_factor]
        all_targets[sample_idx] = ground_truth.cpu().numpy()
        
        # ===== 推理初始化 =====
        inputs = initial_input.clone().to(device)
        predictions = []
        
        # ===== 时间步循环 =====
        for step_idx in tqdm(range(rollout_steps), desc=f"样本 {sample_idx} 时间步", leave=False, position=2):
            # 模型输入 [1, T, C, H, W]
            model_input = inputs.unsqueeze(0).float()
            
            # 模型推理
            with torch.cuda.amp.autocast(enabled=False):  # 混合精度加速
                pred = model(model_input)
            
            # 提取最后一个预测步 [1, 1, C, H, W]
            last_pred = pred[:, -1:]
            
            # 保存预测结果
            pred_np = last_pred.squeeze(0).squeeze(0).cpu().numpy()  # [C, H, W]
            predictions.append(pred_np)
            
            # 更新输入序列（滚动窗口）
            inputs = torch.cat([inputs[1:], last_pred.squeeze(0)], dim=0)
            
            # 可视化保存（每10步）
            if step_idx % 10 == 0:
                true_frame = ground_truth[step_idx].squeeze(0).numpy()
                viz_path = os.path.join(sample_viz_dir, f"step_{step_idx:04d}.png")
                visualize_comparison(pred_np[0], true_frame, step_idx, viz_path)
            
            # 更新进度条
            pbar_total.update(1)
        
        # 保存当前样本结果
        all_outputs[sample_idx] = np.stack(predictions, axis=0)
        
        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================== 结果保存 ==================
print("\n========== 保存结果 ==========")
np.save(os.path.join(result_dir, f"{backbone}_initial_conditions.npy"), all_inputs)
np.save(os.path.join(result_dir, f"{backbone}_predictions.npy"), all_outputs)
np.save(os.path.join(result_dir, f"{backbone}_ground_truth.npy"), all_targets)

# ================== 最终验证 ==================
print("\n========== 结果验证 ==========")
print(f"初始条件形状: {all_inputs.shape}")
print(f"预测结果形状: {all_outputs.shape}")
print(f"真实标签形状: {all_targets.shape}")
print(f"最大值验证 - 初始: {all_inputs.max():.4f}, 预测: {all_outputs.max():.4f}, 真实: {all_targets.max():.4f}")
print(f"最小值验证 - 初始: {all_inputs.min():.4f}, 预测: {all_outputs.min():.4f}, 真实: {all_targets.min():.4f}")

print("\n✅ 推理流程完成！所有结果已保存至:", os.path.abspath(result_dir))