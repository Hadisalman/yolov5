import numpy as np
import torch
from typing import Callable, Optional, Tuple
from dataclasses import replace

from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

from numba import njit
import cv2
import random


'''
fast broadcasted matmul for small matrices borrowed from https://stackoverflow.com/a/59356461
'''
@njit(parallel=False, fastmath=True, inline='always')
def dot_numba(A,B):
    """
    calculate broadcasted matmul on 3x3 matrices with respective shapes: (b,x,y) (b,y,z)
    """
    res=np.empty((A.shape[0],A.shape[1],B.shape[2]),dtype=A.dtype)
    for ii in range(A.shape[0]):
        for i in range(A.shape[1]):
            for j in range(B.shape[2]):
                elt
                for k in range(A.shape[2]):
                    elt += A[ii,i,k]*B[ii,k,j]
                res[ii,i,j] = elt
    return res


@njit(parallel=False, fastmath=True, inline='always')
def fast_rotation_matrix(seed, batch_size, height, width, degrees, translate, scale, shear, perspective):
    np.random.seed(seed)
     # Center
    C = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    C[:, 0, 2] = -width / 2  # x translation (pixels)
    C[:, 1, 2] = -height / 2  # y translation (pixels)
    # Perspective
    P = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    P[:, 2, 0] = np.random.uniform(-perspective, perspective, (batch_size,))  # x perspective (about y)
    P[:, 2, 1] = np.random.uniform(-perspective, perspective, (batch_size,))  # y perspective (about x)
    # Rotation and Scale
    R = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    a = np.random.uniform(-degrees, degrees, (batch_size,))
    s = np.random.uniform(1 - scale, 1 + scale, (batch_size,))
    alpha = s*np.cos(a)
    beta = s*np.sin(a)
    R[:,:2,:] = np.array([[alpha, beta, np.zeros((4,))], [-beta, alpha, np.zeros((4,))]]).transpose((2, 0, 1))
    # Shear
    S = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    S[:, 0, 1] = np.tan(np.random.uniform(-shear, shear) * np.pi / 180)  # x shear (deg)
    S[:, 1, 0] = np.tan(np.random.uniform(-shear, shear) * np.pi / 180)  # y shear (deg)
    # Translation
    T = np.repeat(np.expand_dims(np.eye(3), axis=0), batch_size, axis=0)
    T[:, 0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[:, 1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    rM1 = dot_numba(T, S)
    rM2 = dot_numba(rM1, R)
    rM3 = dot_numba(rM2, P)
    rM = dot_numba(rM3, C)

    return rM, s


@njit(parallel=False, fastmath=True, inline='always')
def fast_label_perspective(labels, rotation_matrix, scale_factor, width, height):
    # Transform label coordinates
    xy = np.ones((labels.shape[0], labels.shape[1] * 4, 3))
    xy[:, :, :2] = labels[:, :, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(labels.shape[0], labels.shape[1] * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = dot_numba(xy, rotation_matrix.T)  # transform
    xy = (xy[:, :, :2] / xy[:, :, 2:3] if perspective else xy[:, :, :2]).reshape(labels.shape[0], labels.shape[1], 8)  # perspective rescale or affine

    # create new boxes
    x = xy[:, :, [0, 2, 4, 6]]
    y = xy[:, :, [1, 3, 5, 7]]
    new = np.concatenate((x.min(2), y.min(2), x.max(2), y.max(2)), axis=1).reshape(labels.shape[0], 4, labels.shape[1]).transpose(0, 2, 1)

    # clip
    new[:, :, [0, 2]] = new[:, :, [0, 2]].clip(0, width)
    new[:, :, [1, 3]] = new[:, :, [1, 3]].clip(0, height)

    # filter box candidates
    box1 = labels[:, :, 2:6].transpose(0, 2, 1) * scale_factor
    box2 = new.transpose(0, 2, 1)
    w1, h1 = box1[:, 2, :] - box1[:, 0, :], box1[:, 3, :] - box1[:, 1, :]
    w2, h2 = box2[:, 2, :] - box2[:, 0, :], box2[:, 3, :] - box2[:, 1, :]
    # 2 =: wh_thr, 100 =: ar_thr, 0.01 =: area_thr, 1e-16 =: eps
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    i = (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.01) & (ar < 100)

    labels = labels[i]
    labels[:, 2:6] = new[i]
    transformed_batch = np.split(labels, splitter.cumsum()[:-1], axis=0)
    new_labels = []
    for arr in transformed_batch:
        arr = np.pad(arr, ((0, labels.shape[1]-arr.shape[0]), (0,0)))
    new_labels.append(arr)
    new_labels = np.array(new_labels)
    return new_labels


class ImageRandomPerspective(Operation):
    '''
    Apply random perspective or affine transformation
    with random uniform values parametrizing rotation angle (degrees), num pixels translation, scale factor, shear factor.
    Operates on raw arrays, not tensors.
    '''
    def __init__(self, degrees: int, translate: float, scale: float, shear: int, perspective: float):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def generate_code(self) -> Callable:
        degrees = self.degrees
        translate = self.translate
        scale = self.scale
        shear = self.shear
        perspective = self.perspective

        parallel_range = Compiler.get_iterable()

        def random_perspective(images, dst, indices):
            seed = indices[-1]
            rM, _ = fast_rotation_matrix(seed, images.shape[0], images.shape[1], images.shape[2], degrees, translate, scale, shear, perspective)
            EYE = np.eye(3)

            for i in parallel_range(images.shape[0]):
                if (rM[i] != EYE).any():
                    if perspective:
                        dst[i] = cv2.warpPerspective(images[i], rM[i], dsize=(images.shape[1], images.shape[2]), borderValue=(114, 114, 114))
                    else:
                        dst[i] = cv2.warpAffine(images[i], rM[i,:2,:], dsize=(images.shape[1], images.shape[2]), borderValue=(114, 114, 114))
                else:
                    dst[i] = images[i]

            return dst

        random_perspective.is_parallel = True
        random_perspective.with_indices = True

        return random_perspective

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class LabelRandomPerspective(Operation):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
    
    def generate_code(self) -> Callable:
        width = self.width
        height = self.height
        degrees = self.degrees
        translate = self.translate
        scale = self.scale
        shear = self.shear
        perspective = self.perspective

        def random_perspective(labels, dst, indices):
            seed = indices[-1]
            rM, s = fast_rotation_matrix(seed, images.shape[0], images.shape[1], images.shape[2], degrees, translate, scale, shear, perspective)
            dst = fast_label_perspective(labels, rM, s, width, height)
            return dst
        
        random_perspective.with_indices = True

        return random_perspective
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class ImageAlbumentation(Operation):
    def __init__(self):
        super().__init__()
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)])

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def generate_code(self) -> Callable:
        transform = self.transform

        parallel_range = Compiler.get_iterator()

        def call_transform(images, dst, indices):
            random.seed(indices[-1])
            if transform is not None:
                for i in parallel_range(images.shape[0]):
                    dst[i] = transform(image=image)["image"]
            else:
                dst = images
            return dst
        
        call_transform.is_parallel = True
        call_transform.with_indices = True

        return call_transform

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class LabelAlbumentation(Operation):
    def __init__(self, imgsz):
        super().__init__()
        self.imgsz = imgsz
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def generate_code(self) -> Callable:
        imgsz = self.imgsz
        transform = self.transform

        parallel_range = Compiler.get_iterator()

        def call_transform(labels, dst, indices):
            random.seed(indices[-1])
            if transform is not None:
                for i in parallel_range(labels.shape[0]):
                    new = transform(image=np.zeros((3, imgsz, imgsz)), bboxes=labels[i][:, 2:], class_labels=labels[i][:, 1])
                    label_arr = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
                    if label_arr.shape[0] < dst.shape[1]:
                        label_arr = np.concatenate([label_arr, np.zeros((dst.shape[1] - label_arr.shape[0], 5))])
                    label_arr = np.concatenate([np.zeros((dst.shape[1], 1)), label_arr], axis=1)
                    dst[i] = label_arr
            else:
                dst = labels
            return dst
        
        call_transform.is_parallel = True
        call_transform.with_indices = True

        return call_transform

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class HSVGain(Operation):
    def __init__(self):
        super().__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    
    def generate_code(self) --> Callable:

        parallel_range = Compiler.get_iterator()

        def augment_hsv(images, dst):

            dst = images
            hsv_dtype = images.dtype  # uint8

            if self.hgain or self.sgain or self.vgain:
                for i in parallel_range(images.shape[0]):
                    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
                    hue, sat, val = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV))

                    x = np.arange(0, 256, dtype=r.dtype)
                    lut_hue = ((x * r[0]) % 180).astype(hsv_dtype)
                    lut_sat = np.clip(x * r[1], 0, 255).astype(hsv_dtype)
                    lut_val = np.clip(x * r[2], 0, 255).astype(hsv_dtype)

                    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
                    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=dst[i])
            
            return dst
        
        augment_hsv.is_parallel = True

        return augment_hsv

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class ImageRandomFlipUD(Operation):
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob
    
    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flipud(images, dst, indices):
            np.random.seed(indices[-1])
            should_flip = np.random.rand(images.shape[0]) < flip_prob
            for i in parallel_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = np.flipud(images[i])
                else:
                    dst[i] = images[i]
            return dst
        
        flipud.is_parallel = True
        flipud.with_indices = True

        return flipud

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class LabelRandomFlipUD(Operation):
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob
    
    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def flipud(labels, dst, indices):
            dst = labels
            np.random.seed(indices[-1])
            should_flip = np.random.rand(labels.shape[0]) < flip_prob
            for i in parallel_range(labels.shape[0]):
                if should_flip[i]:
                    dst[i][:, 3] = 1 - labels[i][:, 3]
            return dst
        
        flipud.is_parallel = True
        flipud.with_indices = True

        return flipud

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class ImageRandomFlipUD(Operation):
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob
    
    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def fliplr(images, dst, indices):
            np.random.seed(indices[-1])
            should_flip = np.random.rand(images.shape[0]) < flip_prob
            for i in parallel_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = np.fliplr(images[i])
                else:
                    dst[i] = images[i]
            return dst
        
        fliplr.is_parallel = True
        fliplr.with_indices = True

        return fliplr

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))


class LabelRandomFlipLR(Operation):
    def __init__(self, flip_prob: float = 0.5):
        super().__init__()
        self.flip_prob = flip_prob
    
    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterator()
        flip_prob = self.flip_prob

        def fliplr(labels, dst, indices):
            dst = labels
            np.random.seed(indices[-1])
            should_flip = np.random.rand(labels.shape[0]) < flip_prob
            for i in parallel_range(labels.shape[0]):
                if should_flip[i]:
                    dst[i][:, 2] = 1 - labels[i][:, 2]
            return dst
        
        fliplr.is_parallel = True
        fliplr.with_indices = True

        return fliplr

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))