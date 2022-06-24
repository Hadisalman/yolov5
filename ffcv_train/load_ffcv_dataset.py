'''
Script for ffcv loading the COCO dataset.
Should be used after writing the COCO dataset to a .beton via write_ffcv_dataset.py
'''

import os
from typing import List

import numpy as np
import torch as ch
import torchvision

from ffcv.fields import RGBImageField, NDArrayField, BytesField, JSONField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, BytesDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, \
    RandomHorizontalFlip, Cutout, RandomTranslate, Convert
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from custom_fields import Variable2DArrayField, CocoShapeField, \
    Variable2DArrayDecoder, CocoShapeDecoder

file_cwd = os.path.dirname(__file__)
base_path = os.path.join(file_cwd, 'datasets')

def load_ffcv_dataset(write_name, batch_size):
    loaders = {}
    for split in ['train', 'val']:
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder(), ToTensor(), ToDevice('cuda:0', non_blocking=True), ToTorchImage(), Convert(ch.uint8)]
        label_pipeline: List[Operation] = [Variable2DArrayDecoder(), ToTensor(), ToDevice('cuda:0')]
        metadata_pipeline: List[Operation] = [BytesDecoder()]
        len_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]

        # Create loaders
        loaders[split] = Loader(base_path + '/' + write_name + '/' + write_name + '_' + split + '.beton',
                                batch_size=batch_size,
                                num_workers=8,
                                order=OrderOption.SEQUENTIAL,
                                drop_last=(split == 'train'),
                                custom_fields={'labels': Variable2DArrayField(second_dim=6, dtype=np.dtype('float64'))},
                                pipelines={'image': image_pipeline,
                                        'labels': label_pipeline,
                                        'metadata': metadata_pipeline,
                                        'labels_len': len_pipeline})
    return loaders

if __name__ == '__main__':
    write_name = 'coco'
    loaders = load_ffcv_dataset(write_name)
    for single_excerpt in loaders['val']:
        print(single_excerpt)
        break