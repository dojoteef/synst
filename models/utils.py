'''
Model utilities
'''

import os
import shutil
import tempfile
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from utils.beam_search import BeamSearchDecoder


def restore(path, modules, num_checkpoints=1, map_location=None, strict=True):
    '''
    Restore from a checkpoint

    Args:
        path - path to restore from
        modules - a dict of name to object that supports the method load_state_dict
    '''
    if not os.path.isfile(path):
        print(f'Cannot find checkpoint: {path}')
        return 0, 0

    print(f'Loading checkpoint {path}')
    state = torch.load(path, map_location=map_location)

    if 'model' in modules:
        model_state = state['model']
        root, ext = os.path.splitext(path)

        # strip any trailing digits
        base = root.rstrip(''.join(str(i) for i in range(10)))

        # determine the integer representation of the trailing digits
        idx = root[len(base):]
        start_idx = int(idx) if idx else 0

        count = 1
        for idx in range(1, num_checkpoints):
            # use the digits as the start index for loading subsequent checkpoints for averaging
            path = f'{base}{start_idx + idx}{ext}'
            if not os.path.isfile(path):
                print(f'Cannot find checkpoint: {path} Skipping it!')
                continue

            print(f'Averaging with checkpoint {path}')
            previous_state = torch.load(path, map_location=map_location)
            previous_model_state = previous_state['model']
            for name, param in model_state.items():
                param.mul_(count).add_(previous_model_state[name]).div_(count + 1)

            count += 1

    for name, obj in modules.items():
        if isinstance(obj, nn.Module):
            obj.load_state_dict(state[name], strict=strict)
        else:
            obj.load_state_dict(state[name])

    return state['epoch'], state['step']


def checkpoint(epoch, step, modules, directory, filename='checkpoint.pt', max_checkpoints=5):
    '''
    Save a checkpoint

    Args:
        epoch - current epoch
        step - current step
        modules - a dict of name to object that supports the method state_dict
        directory - the directory to save the checkpoint file
        filename - the filename of the checkpoint
        max_checkpoints - how many checkpoints to keep
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)

    state = {
        'step': step,
        'epoch': epoch,
    }

    for name, obj in modules.items():
        state[name] = obj.state_dict()

    with tempfile.NamedTemporaryFile() as temp_checkpoint_file:
        torch.save(state, temp_checkpoint_file)

        checkpoint_path = os.path.join(directory, filename)
        if os.path.exists(checkpoint_path):
            root, ext = os.path.splitext(filename)
            for i in range(max_checkpoints - 2, -1, -1):
                previous_path = os.path.join(directory, f'{root}{i}{ext}') if i else checkpoint_path
                if os.path.exists(previous_path):
                    backup_path = os.path.join(directory, f'{root}{i+1}{ext}')
                    if os.path.exists(backup_path):
                        os.replace(previous_path, backup_path)
                    else:
                        os.rename(previous_path, backup_path)

        shutil.copy(temp_checkpoint_file.name, f'{checkpoint_path}.incomplete')
        os.rename(f'{checkpoint_path}.incomplete', checkpoint_path)

    return checkpoint_path


class LabelSmoothingLoss(nn.Module):
    '''
    Implements the label smoothing loss as defined in
    https://arxiv.org/abs/1512.00567

    The API for this loss is modeled after nn..CrossEntropyLoss:

    1) The inputs and targets are expected to be (B x C x ...), where B is the batch dimension, and
    C is the number of classes
    2) You can pass in an index to ignore
    '''
    def __init__(self, smoothing=0.0, ignore_index=-1, reduction='sum'):
        ''' Initialize the label smoothing loss '''
        super(LabelSmoothingLoss, self).__init__()

        self.reduction = reduction
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets): # pylint:disable=arguments-differ
        ''' The implements the actual label smoothing loss '''
        num_classes = inputs.shape[1]
        smoothed = inputs.new_full(inputs.shape, self.smoothing / num_classes)
        smoothed.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        if self.ignore_index >= 0 and self.ignore_index < num_classes:
            smoothed[:, self.ignore_index] = 0.

            mask = targets == self.ignore_index
            smoothed.masked_fill_(mask.unsqueeze(1), 0.)

        return F.kl_div(inputs.log_softmax(1), smoothed, reduction=self.reduction)


class LinearLRSchedule(object):
    '''
    Implements a linear learning rate schedule. Since learning rate schedulers in Pytorch want a
    mutiplicative factor we have to use a non-intuitive computation for linear annealing.

    This needs to be a top-level class in order to pickle it, even though a nested function would
    otherwise work.
    '''
    def __init__(self, initial_lr, final_lr, total_steps):
        ''' Initialize the learning rate schedule '''
        self.initial_lr = initial_lr
        self.lr_rate = (initial_lr - final_lr) / total_steps

    def __call__(self, step):
        ''' The actual learning rate schedule '''
        # Compute the what the previous learning rate should be
        prev_lr = self.initial_lr - step * self.lr_rate

        # Calculate the current multiplicative factor
        return prev_lr / (prev_lr + (step + 1) * self.lr_rate)


class WarmupLRSchedule(object):
    '''
    Implement the learning rate schedule from Attention is All You Need

    This needs to be a top-level class in order to pickle it, even though a nested function would
    otherwise work.
    '''
    def __init__(self, warmup_steps=4000):
        ''' Initialize the learning rate schedule '''
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        ''' The actual learning rate schedule '''
        # the schedule doesn't allow for step to be zero (it's raised to the negative power),
        # but the input step is zero-based so just do a max with 1
        step = max(1, step)
        return min(step ** -0.5, step * self.warmup_steps ** -1.5)


class ModuleWrapper(nn.Module):
    ''' A wrapper module that calls a particular method in it's forward pass '''
    def __init__(self, module, method_name):
        ''' Initializer the wrapper '''
        super(ModuleWrapper, self).__init__()

        self.module = module
        self.method_name = method_name

    def forward(self, *args, **kwargs): # pylint:disable=arguments-differ
        ''' Call the module's method with the passed in parameters '''
        method = getattr(self.module, self.method_name)
        return method(*args, **kwargs)


class Translator(object):
    ''' An object that encapsulates model evaluation '''
    def __init__(self, config, model, dataset):
        self.config = config
        self.dataset = dataset

        self.span = model.span
        self.encoder = ModuleWrapper(model, 'encode')
        self.decoder = ModuleWrapper(model, 'decode')

        self.modules = {
            'model': model
        }

    def to(self, device):
        ''' Move the translator to the specified device '''
        if 'cuda' in device.type:
            self.encoder = nn.DataParallel(self.encoder.cuda())
            self.decoder = nn.DataParallel(self.decoder.cuda())

        return self

    @property
    def sos_idx(self):
        ''' Get the sos index '''
        return self.dataset.sos_idx

    @property
    def eos_idx(self):
        ''' Get the eos index '''
        return self.dataset.eos_idx

    @property
    def padding_idx(self):
        ''' Get the padding index '''
        return self.dataset.padding_idx

    def translate(self, batch):
        ''' Generate with the given batch '''
        with torch.no_grad():
            if self.config.length_basis:
                length_basis = batch[self.config.length_basis]
            else:
                length_basis = [0] * len(batch['inputs'])

            decoder = BeamSearchDecoder(
                self.decoder,
                self.eos_idx,
                self.config,
                self.span
            )

            encoded = self.encoder(batch['inputs'])
            beams = decoder.initialize_search(
                [[self.sos_idx] * self.span for _ in range(len(batch['inputs']))],
                [l + self.config.max_decode_length + self.span + 1 for l in length_basis]
            )
            targets = [
                beam.best_hypothesis.sequence[self.span - 1:]
                for beam in decoder.decode(encoded, beams)
            ]

            gold_targets = []
            gold_target_lens = batch['target_lens']
            for i, target in enumerate(batch['targets']):
                target_len = gold_target_lens[i]
                gold_targets.append(target[:target_len].tolist())

            return OrderedDict([
                ('targets', targets),
                ('gold_targets', gold_targets)
            ])


def get_final_state(x, mask, dim=1):
    ''' Collect the final state based on the passed in mask '''
    shape = list(x.shape)
    dims = len(shape)
    dim = dim % dims

    view_dims = [-1] + [1] * (dims - 1)
    indices_shape = shape[:dim] + [1] + shape[dim + 1:]
    num_padding = mask.sum(1).view(*view_dims).expand(indices_shape)

    return x.gather(dim, shape[dim] - num_padding - 1).squeeze(dim)
