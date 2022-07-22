# `ffcv` Open-Source Object Detection Training: YOLOv5
This is an add-on package for the YOLOv5 object detection model family, containing scripts for writing, loading, and training on ffcv datasets that directly interface with the existing YOLOv5 codebase. Run `train_ffcv_dataset.py` to train YOLOv5 models on the COCO dataset to equivalent performance as YOLOv5's pretrained benchmarks, in only 10-20% of the time compared to the default YOLOv5 datasets.

## Citation
If you use this setup in your research, cite:

```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2022},
    howpublished = {\url{https://github.com/libffcv/ffcv/}},
    note = {commit xxxxxxx}
}
```
(Make sure to replace ``xxxxxxx`` above with the hash of the commit used!)

## Setup

First, clone the YOLOv5 repo via (https://github.com/ultralytics/yolov5), following their installation instructions. Then, navigate into the YOLOv5 directory and clone this repo.

Before training, download a dataset of choice by navigating into the YOLOv5 directory and running the terminal command:
```bash
python -c 'from utils.general import check_dataset, check_file; check_dataset(check_file("FILENAME_OF_DATASET_YAML"))'
```
where `FILENAME_OF_DATASET_YAML` is replaced by the dataset configuration file of choice. For example, to download the COCO dataset as specified in `yolov5/data/coco.yaml`, run the command:
```bash
python -c 'from utils.general import check_dataset, check_file; check_dataset(check_file("coco.yaml"))'
```

Overall, the intended folder structure with datasets downloaded locally is:
```bash
# parent
# ├── datasets
# ├── yolov5
#     └── ffcv_yolov5
```

## Training Loop

With a dataset of choice downloaded locally (as above), you can train models with a single command. Navigate into the ffcv_yolov5 directory and run the command:

```bash
python train_ffcv_dataset.py --data FILENAME_OF_DATASET_YAML.yaml --ffcv-path LABEL_OF_FFCV_DATASETS
```
where `FILENAME_OF_DATASET_YAML` is as in the above setup, and `LABEL_OF_FFCV_DATASETS` is the name of choice for your ffcv datasets. On the first run of any training loop using a given dataset, this command will write ffcv `.beton` files to
- `yolov5/ffcv_yolov5/datasets/[LABEL_OF_FFCV_DATASETS]_train.beton`, and
- `yolov5/ffcv_yolov5/datasets/[LABEL_OF_FFCV_DATASETS]_val.beton`,

respectively, before proceeding to the training loop. On subsequent runs using the same dataset, the existing .beton files will be loaded from as long as the corresponding `LABEL_OF_FFCV_DATASETS` is given in the parameter to the training command.

Additional parameters can be specified in the training command, in case you want to use non-default values. For example, the image size, batch size, number of epochs, initialization for yolov5 weights, and number of workers for ffcv data loading can all be specified:
```bash
python train_ffcv_dataset.py --data FILENAME_OF_DATASET_YAML.yaml --ffcv-path LABEL_OF_FFCV_DATASETS --img 480 --batch 32 --epochs 300 --weights yolov5l.pt --num-workers 12
```
To view all additional parameters and their default values, refer to the `parse_opt()` function in `yolov5/ffcv_yolov5/train_ffcv_dataset.py`.



## Training Details
<p><b>System setup.</b> <!-- TODO --> </p>

<p><b>Dataset setup.</b> To accommodate variable-length labels for bounding boxes in object detection datasets, our ffcv write pipeline contains a custom field, `Variable2DArrayField`, which accommodates bounding boxes. <!-- Full documentation on Variable2DArrayField can be found on the ffcv api here: -->

This custom field allocates memory for each array equal in size to the maximum-length array, so the ffcv loader will read arbitrary data for array values located beyond than the original array's length and within the maximum length. To truncate to the original arrays, and to collate data labels into batches as ingested by YOLOv5 models, we add bounding box label length as an additional data point in our customized YOLOv5 indexed dataset. Refer to the `CocoBoundingBox` class in `yolov5/ffcv_yolov5/write_ffcv_dataset.py` for the indexed dataset.
</p>

## Results
<!-- ImageNet example contains a relevant figure on the right hand side here;
Let's get a similar figure of mAP vs. training time here, for ffcv vs. default YOLOv5 -->

## Configurations
The configurations corresponding to the above results are as follows:

|   mAP[0.5] |   mAP[0.5:0.95] |   # Epochs |   Time (mins) | Architecture   | Setup    |
|--------:|--------:|-----------:|--------------:|:---------------|:---------|
| todo | todo |  todo > 300 |       todo | YOLOv5s      | 8 x A100 |
| todo | todo |         300 |       todo | YOLOv5s      | 8 x A100 |
| todo | todo |  todo < 300 |       todo | YOLOv5s      | 8 x A100 |
| todo | todo |  todo < 300 |       todo | YOLOv5s      | 8 x A100 |
| todo | todo |         300 |       todo | YOLOv5n      | 1 x A100 |
| todo | todo |         300 |       todo | YOLOv5s      | 1 x A100 |
| todo | todo |         300 |       todo | YOLOv5m      | 1 x A100 |
| todo | todo |         300 |       todo | YOLOv5l      | 1 x A100 |
| todo | todo |         300 |       todo | YOLOv5x      | 1 x A100 |

<!-- Can decide on a different configuration structure if necessary -->

<!-- Copy over some benchmark figures from yolo readme -->

## FAQ
<!-- if necessary -->
