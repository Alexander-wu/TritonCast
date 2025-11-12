from model_baselines.fourier_unet import *

# ============== Data loader ==============
from dataloader_ns import SpatioTemporalDataset 


data_path = '/jizhicfs/easyluwu/ocean_project/NPJ_baselines/Exp_6_NS/dataset/McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt'
data = torch.load(data_path)
ns_all = data['vorticity']  # shape: [1280, 100, 128, 128]

# Split the data into training, validation, and test sets
total_samples = ns_all.shape[0]
train_end = int(0.8 * total_samples)
val_end = int(0.9 * total_samples)

train_data = ns_all[:train_end]               # 1024 
val_data = ns_all[train_end:val_end]          # 128 
test_data = ns_all[val_end:]                  # 128 

args = {
    'input_length': data_config['input_length'],
    'target_length': data_config['target_length'],
    'variables_input': data_config.get('variables_input', [0]),
    'variables_output': data_config.get('variables_output', [0]),
    'downsample_factor': data_config['downsample_factor']
}

train_dataset = SpatioTemporalDataset(train_data, args)
val_dataset = SpatioTemporalDataset(val_data, args)
test_dataset = SpatioTemporalDataset(test_data, args)

if parallel_method == 'DistributedDataParallel':
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
else:
    train_sampler = None
    val_sampler = None
    test_sampler = None

train_loader = data_utils.DataLoader(
    train_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=train_sampler,
    shuffle=(train_sampler is None)
)

val_loader = data_utils.DataLoader(
    val_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=val_sampler,
    shuffle=False
)

test_loader = data_utils.DataLoader(
    test_dataset,
    num_workers=0,
    batch_size=training_config['batch_size'],
    sampler=test_sampler,
    shuffle=False
)


if local_rank == 0:
    for input_frames, output_frames in train_loader:
        print(f'Dataloader Input shape: {input_frames.shape}, Output shape: {output_frames.shape}')
        break

# model torch.Size([20, 1, 1, 128, 128]) ; torch.Size([20, 1, 1, 128, 128])
fourier_unet_init_args = {
        "hidden_channels": 32,
        "modes1": 8,
        "modes2": 8,
        "norm": True,
        "n_fourier_layers": 1, 
    }
    
common_args = {
        "n_input_scalar_components": 1,   
        "n_input_vector_components": 0,   
        "n_output_scalar_components": 1, 
        "n_output_vector_components": 0,  
        "time_history": 1,               
        "time_future": 1,                 
        "activation": "silu",            
    }

all_config_args = {**common_args, **fourier_unet_init_args}
model = FourierUnet(**all_config_args)

total_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")
print(f"Parameters in M: {total_params / 1e6:.2f}M")
print(f"Parameters in B: {total_params / 1e9:.2f}B")

B, C, H, W = 20, 1, 128, 128
input_tensor = torch.randn(B, all_config_args["time_history"], C, H, W)
output_tensor = model(input_tensor)
print(input_tensor.shape, output_tensor.shape)