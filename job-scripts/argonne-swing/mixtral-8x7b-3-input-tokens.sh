#!/bin/bash

#SBATCH -J mixtral-8x7b-3-input-tokens

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=6:00:00 
#SBATCH --gres=gpu:3

N_NODES=1
N_GPUS_PER_NODE=3
N_GPUS=$((N_NODES * N_GPUS_PER_NODE))

SYSTEM="argonne-swing"

module load amd-uprof

cd /home/ac.gwilkins/energy-inference/cuda/    

HF_NAME="mistralai/Mixtral-8x7B-v0.1"
MODEL_NAME="mixtral-8x7b"
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")

mkdir -p $MODEL_NAME/$DATE/$TIME
nvidia-smi -lms 100 -f $MODEL_NAME/$DATE/$TIME/nvidia-smi.csv --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv &
pid=$!
AMDuProfCLI timechart --event power --interval 100 --duration 99999 -o ./$MODEL_NAME/$DATE/$TIME/ python3 cuda.py --out_dir ./$MODEL_NAME/$DATE/$TIME --num_tokens 32 --hf_name $HF_NAME --system_name $SYSTEM
kill $pid
