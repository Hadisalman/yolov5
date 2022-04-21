#!/bin/bash
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --time=99:99:00
#SBATCH --output=logs/yolov5_default_benchmark.out
#SBATCH --error=logs/yolov5_default_ERROR.err

python ../train.py --img 640 --batch 16 --epochs 5 --data coco.yaml --weights yolov5s.pt