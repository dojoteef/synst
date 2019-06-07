'''
A module for running data in parallel on multiple devices
'''
import contextlib
from functools import partial

import torch
from torch.nn.parallel import scatter_gather
from torch.nn.parallel._functions import Scatter


# NOTE: chunked_scatter is a copy of torch.nn.parallel.scatter_gather.scatter that has been modified
# to account for chunk_sizes (which Scatter.apply already supports)
def chunked_scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks or specified chunk sizes (based on the
    'chunk_sizes' key in a dict) and distributes them across given GPUs. Duplicates references to
    objects that are not tensors.

    Example::
        >>> target_gpus=[0, 1]
        >>> inputs = [{
            'chunk_sizes': [5, 2],
            'tensor1': torch.ones(7, 3),
            'nested1': {
                'chunk_sizes': [3, 4],
                'tensor2': 2 * torch.ones(7, 3)
            },
            'nested2': {
                'tensor3': 3 * torch.ones(7, 3)
            }
        }]
        >>> chunked_scatter(inputs, target_gpus)
        [
         [{
           'tensor1': tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.],
                              [1., 1., 1.], [1., 1., 1.]], device='cuda:0'),
           'chunk_sizes': [5, 2],
           'nested1': {
            'chunk_sizes': [3, 4],
            'tensor2': tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]], device='cuda:0')
           },
           'nested2': {
            'tensor3': tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.],
                               [3., 3., 3.], [3., 3., 3.]], device='cuda:0')
           }
         }],
         [{
           'tensor1': tensor([[1., 1., 1.], [1., 1., 1.]], device='cuda:1'),
           'chunk_sizes': [5, 2],
           'nested1': {
             'chunk_sizes': [3, 4],
             'tensor2': tensor([[2., 2., 2.], [2., 2., 2.],
                                [2., 2., 2.], [2., 2., 2.]], device='cuda:1')
           },
           'nested2': {
            'tensor3': tensor([[3., 3., 3.], [3., 3., 3.]], device='cuda:1')
           }
         }]
        ]

    .. warning:: If specified, attr:`chunk_sizes` must equal to the number of target gpus.
    """
    def scatter_map(obj, chunk_sizes=None):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        if isinstance(obj, tuple) and obj:
            chunked_scatter_map = partial(scatter_map, chunk_sizes=chunk_sizes)
            return list(zip(*map(chunked_scatter_map, obj)))
        if isinstance(obj, list) and obj:
            chunked_scatter_map = partial(scatter_map, chunk_sizes=chunk_sizes)
            return list(map(list, zip(*map(chunked_scatter_map, obj))))
        if isinstance(obj, dict) and obj:
            chunk_sizes = obj.get('chunk_sizes', chunk_sizes)
            chunked_scatter_map = partial(scatter_map, chunk_sizes=chunk_sizes)
            return list(map(type(obj), zip(*map(chunked_scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


@contextlib.contextmanager
def chunked_scattering():
    ''' A context manager that monkey patches scatter to support chunk_sizes '''
    old_scatter = scatter_gather.scatter
    scatter_gather.scatter = chunked_scatter
    yield
    scatter_gather.scatter = old_scatter
