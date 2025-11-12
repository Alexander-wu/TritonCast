#  dataloader_ns.py

import torch
import torch.utils.data as data_utils

class SpatioTemporalDataset(data_utils.Dataset):
    """
    用于时空数据预测的通用数据集类。
    它接收一个预先分割好的数据张量，并使用滑动窗口方法生成 (输入, 目标) 样本对。
    """
    def __init__(self, data, args):
        """
        初始化数据集。
        Args:
            data (torch.Tensor): 形状为 [样本数, 时间步数, H, W] 的数据张量。
                                 这应该是已经分割好的训练/验证/测试数据。
            args (dict): 包含模型和数据参数的字典。
                         需要 'input_length', 'target_length', 'downsample_factor'。
        """
        super(SpatioTemporalDataset, self).__init__()
        self.args = args
        self.input_length = args['input_length']
        self.target_length = args['target_length']
        self.downsample_factor = args.get('downsample_factor', 1) # 使用 .get 提供默认值，更安全

        # 假设传入的数据已经是最终要使用的数据，我们只为其增加一个“变量”维度。
        # 原始形状: [样本数, 时间步数, H, W]
        # 目标形状: [样本数, 时间步数, 1, H, W]
        self.data = data.unsqueeze(2)

        # --- 核心修改：移除了内部的数据分割逻辑 ---
        # self.start_index = ...
        # self.end_index = ...
        # self.data = self.data[self.start_index:self.end_index]
        # ---------------------------------------------

        self.num_samples = self.data.shape[0]
        self.num_time_steps = self.data.shape[1]
        self.variables_input = args.get('variables_input', [0])
        self.variables_output = args.get('variables_output', [0])

        # 创建样本索引（滑动窗口逻辑保持不变）
        self.sample_indices = []
        # 计算一个时间序列中可以创建多少个样本
        max_start_time = self.num_time_steps - self.input_length - self.target_length + 1
        
        for s in range(self.num_samples):
            # t 代表的是输入序列结束、目标序列开始的时间点
            for t_start in range(max_start_time):
                t = t_start + self.input_length
                self.sample_indices.append((s, t))

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """根据索引 idx 获取一个 (输入, 目标) 样本对。"""
        s, t = self.sample_indices[idx]

        # 提取输入和目标序列
        # 输入序列: [t - input_length, t)
        input_seq = self.data[s, t - self.input_length:t, self.variables_input, :, :]
        # 目标序列: [t, t + target_length)
        target_seq = self.data[s, t:t + self.target_length, self.variables_output, :, :]

        # 如果需要，进行空间下采样
        dsf = self.downsample_factor
        if dsf > 1:
            input_seq = input_seq[:, :, ::dsf, ::dsf]
            target_seq = target_seq[:, :, ::dsf, ::dsf]

        # 确保数据类型为 float
        input_seq = input_seq.float()
        target_seq = target_seq.float()

        return input_seq, target_seq