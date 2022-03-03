'''
Script for writing an FFCV dataset from the Microsoft COCO detection dataset.
Requires the COCO API to be downloaded; consult https://github.com/cocodataset/cocoapi for instructions for downloading the pycocotools package.
'''

import numpy as np
import math
import os
import sys
import glob
import cv2
import hashlib
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from PIL import ExifTags, Image, ImageOps
from argparse import ArgumentParser

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField, BytesField, JsonField
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset, Subset
import torch

from custom_fields import Variable2DArrayField, StringField, CocoShapeField

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # multithreading dataset caching

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class CocoBoundingBox(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', label_pad=False):
        '''
        REMOVED:
        batch size, augmentations, hyperparameters, image weights, rect (vs. square pad), Mosaic
        '''
        self.path = path
        self.img_size = img_size
        self.augment = None
        self.label_pad = label_pad

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')

        '''
        INCLUDED: cache for image/label pairs
        The YOLOV5 pipeline only accesses labels through reading or generating a cache so it's easier to keep this in place
        '''

        # Check cache
        self.label_files = DatasetUtils.img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == DatasetUtils.get_hash(self.label_files + self.img_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'
        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = DatasetUtils.img2label_paths(cache.keys())  # update
        self.n = len(shapes) # number of images
        self.indices = range(self.n)

        '''
        EXCLUDED: single-class training or filter to only certain classes
        '''
        self.single_cls = None
        self.included_classes = None
        if self.single_cls or self.included_classes:
            pass

        '''
        EXCLUDED: Load rectangular images, no square padding
        '''
        self.rect = None
        if self.rect:
            pass

        '''
        EXCLUDED: Cache images for downstream dataloader; ffcv will do this instead
        '''

        '''
        Calculate max length labels for padding
        '''
        self.max_label = None
        if self.label_pad:
            self.max_label = max([self.__getitem__(i)[1].shape[0] for singlet in range(self.n)])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        '''
        EXCLUDED: Mosaic & MixUp augmentations
        Later: implement in ffcv pipeline transforms
        '''

        # Load image
        # img, (h0, w0), (h, w) = self.load_image(index)
        with Image.open(self.img_files[index]) as im:
            img = im
        (w0, h0) = img.size
        r = self.img_size / max(h0, w0)
        (w, h) = (int(w0 * r), int(h0 * r))
        ratio = (r, r)
        pad = ((self.img_size - w)/2, (self.img_size - h)/2)

        # Letterbox
        # img, ratio, pad = DatasetUtils.letterbox(img, self.img_size, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad) # for COCO mAP rescaling

        '''
        !! seems to resize and pad, this should be taken out later
        '''
        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = DatasetUtils.xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        '''
        EXCLUDED: linear transformations (center, perspective, rotation, scale, shear, translation)
        '''
        if self.augment:
            pass

        nl = len(labels)  # number of labels
        #if nl:
        #    labels[:, 1:5] = DatasetUtils.xyxy2xywhn(labels[:, 1:5], w=self.img_size, h=self.img_size, clip=True, eps=1E-3)

        '''
        EXCLUDED: Albumentation, HSV color, flip, cutout augmentations
        '''
        if self.augment:
            pass

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        if self.label_pad:
            labels_out = np.vstack(labels_out, torch.zeros((self.max_label-nl,6)))

        # Convert
        #img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        #img = np.ascontiguousarray(img)

        #return torch.from_numpy(img), labels_out, self.img_files[index], shapes
        return img, labels_out, self.img_files[index], shapes

    '''
    UNUSED: translation from image to tensor is delayed to ffcv pipeline instead
    '''
    def load_image(self, i):
        # loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.img_files[i]
        im = cv2.imread(f)  # BGR
        assert im is not None, f'Image Not Found {f}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im,
                            (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    
    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(DatasetUtils.verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        x['hash'] = DatasetUtils.get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        except Exception as e:
            pass
        return x



class DatasetUtils:

    @staticmethod
    def img2label_paths(img_paths):
        # Define label paths as a function of image paths
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    @staticmethod
    def verify_image_label(args):
        # Verify one image-label pair
        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        try:
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = DatasetUtils.exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                        msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any([len(x) > 8 for x in lb]):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = segments[i]
                        msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)
            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    @staticmethod
    def exif_size(img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            # Get orientation exif tag
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                s = (s[1], s[0])
            elif rotation == 8:  # rotation 90
                s = (s[1], s[0])
        except Exception:
            pass

        return s
    
    @staticmethod
    def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y

    @staticmethod
    def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        if clip:
            DatasetUtils.clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
        y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
        y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
        return y

    @staticmethod
    def clip_coords(boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
    @staticmethod
    def get_hash(paths):
        # Returns a single hash value of a list of paths (files or dirs)
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.md5(str(size).encode())  # hash sizes
        h.update(''.join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash



def parse_opt(known=False):
    parser = ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='train, test, or val set')
    parser.add_argument('--data-dir', type=str, default=ROOT / '../datasets/coco-box', help='Where to find the COCO dataset, directory containing images, labels, and annotations')
    parser.add_argument('--write-name', type=str, default=ROOT / 'default_coco_box', help='Where to write the new dataset')
    parser.add_argument('--write-mode', type=str, default='jpg', help='Mode: raw, smart, proportion, or jpg')
    parser.add_argument('--max-resolution', type=int, default=640, help='Max image side length')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of workers to use')
    parser.add_argument('--chunk-size', type=int, default=100, help='Chunk size for writing')
    parser.add_argument('--jpeg-quality', type=float, default=90, help='Quality of jpeg images')
    parser.add_argument('--subset', type=int, default=-1, help='How many images to use (-1 for all)')
    parser.add_argument('--compress-probability', type=float, default=None, help='Probability of compression; proportion of raw to compress to jpg')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    split, data_dir, write_name, max_resolution, num_workers, chunk_size, subset, jpeg_quality, write_mode, compress_probability = \
        opt.split, Path(opt.data_dir), Path(opt.write_name), opt.max_resolution, opt.num_workers, \
        opt.chunk_size, opt.subset, opt.jpeg_quality, opt.write_mode, opt.compress_probability

    path = '/mnt/nfs/home/branhung/src/datasets/coco-box/train2017.txt'
    imgsz = 640
    label_pad = True
    dataset = CocoBoundingBox(path, imgsz, label_pad = label_pad)
    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    '''
    labels field A: custom BytesField of NDArrays for variable length (k, const) 2d arrays
    labels field B: max length NDArray with padding
    file field A: JsonField
    file field B: BytesField with numpy string <-> char array conversion
    file field C: custom string field
    shapes field A: BytesField with numpy flatten <-> unflatten
    shapes field B: custom int nested tuples field
    '''
    #A
    default_writer = DatasetWriter('/datasets/CUSTOM_' + write_name + '.beton', {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'labels': Variable2DArrayField(),
        'file': JsonField(),
        'shapes': BytesField(),
    }, num_workers=num_workers)

    #B
    default_writer = DatasetWriter('/datasets/default_' + write_name + '.beton', {
        'image': RGBImageField(write_mode=write_mode,
                               max_resolution=max_resolution,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'labels': NDArrayField(),
        'file': BytesField(),
        'shapes': CocoShapeField(),
    }, num_workers=num_workers)

    unused_file_field = StringField()

if __name__ == '__main__':
    path = '/mnt/nfs/home/branhung/src/datasets/coco-box/train2017.txt'
    imgsz = 640
    dataset = CocoBoundingBox(path, imgsz, label_pad = label_pad)
    dataset[0][0].show()
    for i in range(20):
        pprint(dataset[i][1:])


