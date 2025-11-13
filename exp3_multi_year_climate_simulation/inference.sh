

prediction_length=2501 # 31
decorrelation_time=30
n_samples_per_year=365

exp_dir='./exp'
config='Triton' #GraphCOAF GraphCast Masked_AE_Ocean Triton OneForecast

run_num='20250803-152259'
finetune_dir=''
# finetune_dir='2_steps_finetune_new'
year=2020
ics_type='default' # options: default, datetime
date_strings="01/01/${year}-00:00:00,01/02/${year}-00:00:00,01/03/${year}-00:00:00,01/04/${year}-00:00:00,01/05/${year}-00:00:00,01/06/${year}-00:00:00,01/07/${year}-00:00:00,01/08/${year}-00:00:00,01/09/${year}-00:00:00,01/10/${year}-00:00:00,01/11/${year}-00:00:00,01/12/${year}-00:00:00"


CUDA_VISIBLE_DEVICES=0 python inference.py --exp_dir=${exp_dir} --config=${config} --run_num=${run_num} --finetune_dir=$finetune_dir --prediction_length=${prediction_length} --decorrelation_time=${decorrelation_time} --n_samples_per_year=${n_samples_per_year} --ics_type=${ics_type} --date_strings=${date_strings} --year=${year} 



