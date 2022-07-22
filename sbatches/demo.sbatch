#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=99:99:00
#SBATCH --partition=high-priority
#SBATCH --output=logs/yolov5_default_benchmark.out
#SBATCH --error=logs/yolov5_default_ERROR.err

python ../train.py --data coco.yaml --epochs 300 --weights yolov5n.pt --img 640 --batch 16