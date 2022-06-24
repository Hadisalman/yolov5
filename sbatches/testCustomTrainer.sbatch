#!/bin/bash
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:99:00
#SBATCH --output=../sbatches/logs/ffcv_custom.out
#SBATCH --error=../sbatches/logs/ffcv_ERROR.err

python ../ffcv_train/train_ffcv_dataset.py --img 640 --batch 16 --epochs 5 --data coco.yaml --weights yolov5s.pt