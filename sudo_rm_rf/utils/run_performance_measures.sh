#!/bin/bash
#set -e  # Exit on error

out_dir=../../performance_outputs
mkdir -p ${out_dir}
batch_size=1
cuda_device=0
input_samples=8000

# Extract simple measurements
for dev in cpu gpu; do
  for model in baseline_dprnn baseline_demucs baseline_twostep \
    sudormrf_R16 sudormrf_R8 sudormrf_R4 baseline_original_convtasnet; do \
       /usr/bin/python3.6 extract_model_performance.py \
      --device $dev --model_type $model --input_samples $input_samples \
      --batch_size $batch_size -cad $cuda_device --repeats 20 \
      --run_all \
      >> $out_dir/${model}_${dev}_bs_${batch_size}_samples_${input_samples} \
      && \
      /usr/bin/time -v /usr/bin/python3.6 extract_model_performance.py \
      --device $dev --model_type $model --input_samples $input_samples \
      --batch_size $batch_size -cad $cuda_device --repeats 1 \
      --measure forward 2> \
      $out_dir/forwardCPURAM${model}_${dev}_bs_${batch_size}_samples_${input_samples} \
      && /usr/bin/time -v /usr/bin/python3.6 extract_model_performance.py \
      --device $dev --model_type $model --input_samples $input_samples \
      --batch_size $batch_size -cad $cuda_device --repeats 1 \
      --measure backward 2> \
      $out_dir/backwardCPURAM${model}_${dev}_bs_${batch_size}_samples_${input_samples} \
      && echo Processed $model with batch $batch_size and \
      $input_samples samples on $dev;
  done;
done

#batch_size=4
#input_samples=32000
#
## Extract simple measurements
#for dev in cpu gpu; do
#  for model in baseline_dprnn baseline_demucs baseline_twostep \
#    sudormrf_R16 sudormrf_R8 sudormrf_R4 baseline_original_convtasnet; do \
#      /usr/bin/python3.6 extract_model_performance.py --device $dev \
#      --model_type $model --input_samples $input_samples \
#      --batch_size $batch_size -cad $cuda_device --repeats 20 \
#      --run_all >> $out_dir/${model}_${dev}_bs_${batch_size}_samples_${input_samples} \
#      && echo Processed $model with batch $batch_size and
#      $input_samples samples on $dev;
#  done;
#done

