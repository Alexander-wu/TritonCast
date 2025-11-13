prediction_length=121

exp_dir='./exp'
config='Triton' 
run_num='20250819-012814'
finetune_dir='7_steps_finetune'

ics_type='default'

CUDA_VISIBLE_DEVICES=1 python inference_simulation.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



