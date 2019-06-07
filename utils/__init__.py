'''
Various utilities
'''
import contextlib
import io
import random
import sys
import threading
from itertools import tee, zip_longest
from subprocess import check_output, CalledProcessError

import numpy as np
import torch
from tqdm import tqdm


INF = float('inf')
NEG_INF = float('-inf')


# pylint:disable=line-too-long
def ceildiv(x, y):
    ''' https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python#17511341 '''
    return -(-x // y)
# pylint:enable=line-too-long


def pairwise(iterable, longest=False):
    '''
    See itertools recipes:
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    '''
    x, y = tee(iterable)
    next(y, None)
    zip_func = zip_longest if longest else zip
    return zip_func(x, y)


def grouper(iterable, n, fillvalue=None, padded=False):  # pylint:disable=invalid-name
    '''
    See itertools recipes:
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    Collect data into fixed-length chunks or blocks
    '''
    args = [iter(iterable)] * n
    groups = zip_longest(*args, fillvalue=fillvalue)
    if padded:
        # keep the fill value
        return groups
    else:
        # ignore the fill value
        return [[x for x in group if x is not fillvalue] for group in groups]


def partition(seq, num):
    ''' Partition a sequence into num equal parts (potentially except for the last slice) '''
    return [seq[i:i + num] for i in range(0, len(seq), num)]


def divvy(num, chunks):
    ''' Divvy a number into an array of equal sized chunks '''
    chunk_mod = (num % chunks)
    chunk_size = num // chunks
    return [chunk_size + 1] * chunk_mod + [chunk_size] * (chunks - chunk_mod)


def triu(inputs, diagonal=0, span=1, stride=1, offset=0):
    '''
    Returns an upper triangular matrix, but allows span which determines how many contiguous
    elements of the matrix to consider as a "single" number, e.g.

    >> triu(torch.full((8, 8), float('-inf'), 1, 2)
    tensor([[0.0000, 0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
            [0.0000, 0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
            [0.0000, 0.0000, 0.0000, 0.0000,   -inf,   -inf,   -inf,   -inf],
            [0.0000, 0.0000, 0.0000, 0.0000,   -inf,   -inf,   -inf,   -inf],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,   -inf,   -inf],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,   -inf,   -inf],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    '''
    for i, row in enumerate(inputs):
        row[:span * (diagonal + i // stride) + offset] = 0.

    return inputs


def get_version_string():
    ''' Return a git version string for the repo '''
    try:
        version = check_output(['git', 'describe', '--always', '--dirty'], encoding='utf-8')
    except CalledProcessError:
        raise RuntimeError('Call to "git describe" failed!')

    return version


def to_numpy_dtype(dtype):
    ''' Convert a torch dtype to a numpy dtype '''
    return np.dtype(dtype.__reduce__().replace('torch.', ''))


def left_pad(x, dim=-1, count=1, fill=0):
    ''' left pad the given tensor '''
    if not count:
        return x

    shape = list(x.shape)
    dims = len(shape)
    dim = dim % dims
    fill_shape = shape[:dim] + [count] + shape[dim + 1:]
    return torch.cat((x.new_full(fill_shape, fill), x), dim)


def right_pad(x, dim=-1, count=1, fill=0):
    ''' right pad the given tensor '''
    if not count:
        return x

    shape = list(x.shape)
    dims = len(shape)
    dim = dim % dims
    fill_shape = shape[:dim] + [count] + shape[dim + 1:]
    return torch.cat((x, x.new_full(fill_shape, fill)), dim)


def left_shift(x, dim=-1, shift=1, fill=None):
    ''' left shift the given tensor '''
    if not shift:
        return x

    if fill is not None:
        x = right_pad(x, dim, shift, fill)

    shape = list(x.shape)
    dims = len(shape)
    dim = dim % dims
    return x[tuple(slice(shift if d == dim else 0, s + shift) for d, s in enumerate(shape))]


def right_shift(x, dim=-1, shift=1, fill=None):
    ''' Right shift the given tensor '''
    if not shift:
        return x

    if fill is not None:
        x = left_pad(x, dim, shift, fill)

    shape = list(x.shape)
    dims = len(shape)
    dim = dim % dims
    return x[tuple(slice(-shift if d == dim else s) for d, s in enumerate(shape))]


def same_tensor(tensor, *args):
    ''' Do the input tensors all point to the same underlying data '''
    for other in args:
        if not torch.is_tensor(other):
            return False

        if tensor.device != other.device:
            return False

        if tensor.dtype != other.dtype:
            return False

        if tensor.data_ptr() != other.data_ptr():
            return False

    return True


class TQDMStreamWrapper(io.IOBase):
    ''' A wrapper around an existing IO stream to funnel to tqdm '''
    def __init__(self, stream):
        ''' Initialize the stream wrapper '''
        super(TQDMStreamWrapper, self).__init__()
        self.stream = stream

    def write(self, line):
        ''' Potentially write to the stream '''
        if line.rstrip(): # avoid printing empty lines (only whitespace)
            tqdm.write(line, file=self.stream)


_STREAMS = threading.local()
_STREAMS.stdout_stack = []
@contextlib.contextmanager
def tqdm_wrap_stdout():
    ''' Wrap a sys.stdout and funnel it to tqdm.write '''
    _STREAMS.stdout_stack.append(sys.stdout)
    sys.stdout = TQDMStreamWrapper(sys.stdout)
    yield
    sys.stdout = _STREAMS.stdout_stack.pop()


@contextlib.contextmanager
def tqdm_unwrap_stdout():
    ''' Unwrap a tqdm.write and funnel it to sys.stdout '''
    saved = sys.stdout
    sys.stdout = _STREAMS.stdout_stack.pop()
    yield
    _STREAMS.stdout_stack.append(sys.stdout)
    sys.stdout = saved


# Recursively split or chunk the given data structure. split_or_chunk is based on
# torch.nn.parallel.scatter_gather.scatter
def split_or_chunk(inputs, num_chunks_or_sections, dim=0):
    r"""
    Splits tensors into approximately equal chunks or specified chunk sizes (based on the
    'num_chunks_or_sections'). Duplicates references to objects that are not tensors.
    """
    def split_map(obj):
        if isinstance(obj, torch.Tensor):
            if isinstance(num_chunks_or_sections, int):
                return torch.chunk(obj, num_chunks_or_sections, dim=dim)
            else:
                return torch.split(obj, num_chunks_or_sections, dim=dim)
        if isinstance(obj, tuple) and obj:
            return list(zip(*map(split_map, obj)))
        if isinstance(obj, list) and obj:
            return list(map(list, zip(*map(split_map, obj))))
        if isinstance(obj, dict) and obj:
            return list(map(type(obj), zip(*map(split_map, obj.items()))))
        if isinstance(num_chunks_or_sections, int):
            return [obj for chunk in range(num_chunks_or_sections)]
        else:
            return [obj for chunk in num_chunks_or_sections]

    # After split_map is called, a split_map cell will exist. This cell
    # has a reference to the actual function split_map, which has references
    # to a closure that has a reference to the split_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return split_map(inputs)
    finally:
        split_map = None


# Recursively split or chunk the given data structure. split_or_chunk is based on
# torch.nn.parallel.scatter_gather.gather
def cat(outputs, dim=0):
    r"""
    Concatenates tensors recursively in collections.
    """
    def cat_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return torch.cat(outputs, dim=dim)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, cat_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(cat_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return cat_map(outputs)
    finally:
        cat_map = None


def get_random_seed_fn(seed, cuda=True):
    ''' Return a function that sets a random seed '''
    def set_random_seed(worker_id=0): # pylint:disable=unused-argument
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    return set_random_seed
