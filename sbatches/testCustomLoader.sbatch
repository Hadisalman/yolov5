#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=../sbatches/logs/ffcv_custom.out
#SBATCH --error=../sbatches/logs/ffcv_ERROR.err

python ../ffcv/load_coco.py