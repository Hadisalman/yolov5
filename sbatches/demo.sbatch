#!/bin/bash
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --time=99:99:00
#SBATCH --output=logs/yolov5_default_benchmark.out
#SBATCH --error=logs/yolov5_default_ERROR.err

python ../train.py --data coco.yaml