#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:99:00
#SBATCH --partition=high-priority
#SBATCH --output=../sbatches/logs/hadi_ffcv_custom.out
#SBATCH --error=../sbatches/logs/hadi_ffcv_ERROR.err

python ../ffcv_yolov5/train_ffcv_dataset.py --img 640 --batch 16 --epochs 300 --data coco.yaml --weights yolov5n.pt --ffcv-path hadi_coco