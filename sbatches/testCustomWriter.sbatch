#!/bin/bash
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --output=../sbatches/logs/ffcv_custom.out
#SBATCH --error=../sbatches/logs/ffcv_ERROR.err

python ../ffcv_train/write_ffcv_dataset.py