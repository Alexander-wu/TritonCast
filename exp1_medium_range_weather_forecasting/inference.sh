
prediction_length=41

exp_dir='./exp'
config='Triton'
run_num='20250731-162648'
# finetune_dir='11_steps_finetune'
finetune_dir='60_steps_finetune'
ics_type='default'

CUDA_VISIBLE_DEVICES=5 python inference.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --ics_type=${ics_type}



