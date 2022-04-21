'''
Script for ffcv loading the COCO dataset.
Should be used after writing the COCO dataset to a .beton via write_ffcv_dataset.py
'''

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

BATCH_SIZE = 1

def load_ffcv_dataset(write_name):
    loaders = {}
    loaders['nc'] = 80 # BAD MAGIC NUMBER, this should be obtained from the written dataset itself
    for split in ['train','test','val']:
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder(), ToTensor(), ToDevice('cuda:0', non_blocking=True), ToTorchImage(), Convert(ch.uint8)]
        label_pipeline: List[Operation] = [Variable2DArrayDecoder(), ToTensor(), ToDevice('cuda:0')]
        url_pipeline: List[Operation] = [BytesDecoder()]
        shape_pipeline: List[Operation] = [BytesDecoder()]

        # Create loaders
        loaders[split] = Loader('/mnt/nfs/home/branhung/src/yolov5/ffcv/datasets/' + write_name + '/' + write_name + '_' + split + '.beton',
                                batch_size=BATCH_SIZE,
                                num_workers=8,
                                order=OrderOption.SEQUENTIAL,
                                drop_last=(split == 'train'),
                                custom_fields={'labels': Variable2DArrayField(second_dim=6, dtype=np.dtype('float64'))},
                                pipelines={'image': image_pipeline,
                                        'labels': label_pipeline,
                                        'file': url_pipeline,
                                        'shapes': shape_pipeline})
    return loaders

if __name__ == '__main__':
    write_name = 'coco_box'
    loaders = load_ffcv_dataset(write_name)
    for single_excerpt in loaders['val']:
        print(single_excerpt)
        break

    '''
    TODO: check the format of the image load
    TODO: check the effect of json unpack and extract the string
    '''