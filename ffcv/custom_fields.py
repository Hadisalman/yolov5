from typing import Callable, TYPE_CHECKING, Tuple, Type
import json
from dataclasses import replace

import numpy as np

from ffcv.fields.base import Field, ARG_TYPE
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.libffcv import memcpy

from ffcv.fields import BytesField
from ffcv.fields.bytes import BytesDecoder
from ffcv.fields.basics import BasicDecoder

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManage

'''
Variable 2D Array appears to work!
(may require more testing)
'''

class Variable2DArrayDecoder(Operation):

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        max_len = self.metadata['len'].max()

        my_shape = (max_len, self.field.second_dim)
        return (
            replace(previous_state, jit_mode=True,
                    shape=my_shape,
                    dtype=self.field.dtype),
            AllocationQuery(my_shape, dtype=self.field.dtype)
        )


    def generate_code(self) -> Callable:

        ## how are we getting from the max_len from the allocationquery trimmed down to the actual length for each array?

        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()

        def decoder(indices, destination, metadata, storage_state):
            for ix in my_range(indices.shape[0]):
                sample_id = indices[ix]
                data = mem_read(metadata[sample_id]['ptr'], storage_state)
                my_memcpy(data, destination[ix])
            return destination

        return decoder

Variable2DArrayArgsType = np.dtype([
    ('second_dim', '<u8'),  # fixed length of 2nd dimension of 2d np array
    ('type_length', '<u8'),  # length of the dtype description
])

class Variable2DArrayField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting variable-length 2D numpy arrays of any dtype.

    Intended for use with certain np.array data types (i.e., bounding box labels for object detection)
    which have variable length across data samples.

    The writer expects to be passed a 2D numpy array of shape (k, c) for each sample,
    where k is variable and c is constant across the dataset.
    """
    def __init__(self, second_dim, dtype):
        self.second_dim = second_dim
        self.dtype = dtype

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('len', '<u8')
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        header_size = Variable2DArrayArgsType.itemsize
        header = binary[:header_size].view(Variable2DArrayArgsType)[0]
        type_length = header['type_length']
        type_data = binary[header_size:][:type_length].tobytes().decode('ascii')
        type_desc = json.loads(type_data)
        type_desc = [tuple(x) for x in type_desc]
        assert len(type_desc) == 1
        dtype = np.dtype(type_desc)['f0']
        second_dim = int(header['second_dim'])
        return Variable2DArrayField(second_dim, dtype)


    def to_binary(self) -> ARG_TYPE:
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        header = np.zeros(1, dtype=Variable2DArrayArgsType)
        header['second_dim'][0] = np.int64(self.second_dim) ## is the cast to int64 necessary?
        encoded_type = json.dumps(self.dtype.descr)
        encoded_type = np.frombuffer(encoded_type.encode('ascii'), dtype='<u1')
        header['type_length'][0] = len(encoded_type)
        to_write = np.concatenate([header.view('<u1'), encoded_type]) ## should header.view('<u1') be changed because of slightly different structure of second_dim vs. previous ndarray's shape?
        result[0][:to_write.shape[0]] = to_write
        return result


    def encode(self, destination, field, malloc):
        ptr, buffer = malloc(field.size*self.dtype.itemsize)
        buffer[:] = field.reshape(-1).view('<u1')
        destination['ptr'] = ptr
        destination['len'] = field.shape[0]


    def get_decoder_class(self) -> Type[Operation]:
        return Variable2DArrayDecoder

'''
class StringDecoder(BasicDecoder):

    ## Will the basic decoder work?
    ## This was used for fixed length int or float data but the strings are variable length now

    max_len = self.metadata['len'].max()
    dtype = np.dtype(('U', max_len))

class StringField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting string values of variable length
    """
    def __init__(self):
        pass

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('len', '<u8')
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return StringField()


    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]


    def encode(self, destination, field, malloc):
        destination[0] = field
        destination['len'] = len(field)


    def get_decoder_class(self) -> Type[Operation]:
        return StringDecoder
'''

'''
CocoShape can successfully encode but torch can't decode because of the nature of the dtype.
The dtype has a nested structure so that numpy must treat its elements as objects (i.e. np.void type)
but torch cannot convert an ndarray of np.void into a tensor, which is required for any custom field so far
'''

CocoDType = np.dtype([
    ('dim', '<u8', (2,)),
    ('ratio_pad', '<f8', (2,2))
])

class CocoShapeDecoder(BasicDecoder):
    dtype = CocoDType

class CocoShapeField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting (scalar) floating-point (float64)
    values.
    """
    def __init__(self):
        pass

    @property
    def metadata_type(self) -> np.dtype:
        return CocoDType

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return CocoShapeField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]


    def encode(self, destination, field, malloc):
        destination[0] = field


    def get_decoder_class(self) -> Type[Operation]:
        return CocoShapeDecoder
