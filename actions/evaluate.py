'''
SynST

--
Main entry point for evaluating SynST
'''

from __future__ import print_function

import os
import sys
import signal
import time
import atexit
from contextlib import ExitStack

import torch
from torch import nn
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import metrics
from models.utils import restore
from utils import profile
from utils import tqdm_wrap_stdout


class CheckpointEventHandler(FileSystemEventHandler):
    ''' A filesystem event handler for new checkpoints '''
    def __init__(self, handler, experiment, verbose=0):
        ''' Initialize the CheckpointEventHandler '''
        super(CheckpointEventHandler, self).__init__()
        self.watches = set()
        self.handler = handler
        self.verbose = verbose
        self.experiment = experiment

    def on_created(self, event):
        ''' Watcher for a new file '''
        root, ext = os.path.splitext(event.src_path)
        basename = os.path.basename(root)
        if ext == '.incomplete' and basename == 'checkpoint.pt':
            self.watches.add(event.src_path)

            if self.verbose > 1:
                print(f'Waiting for {event.src_path}')

    def on_moved(self, event):
        ''' Handle when a file has been modified '''
        if event.src_path in self.watches:
            self.watches.remove(event.src_path)
            self.handler(event.dest_path, self.experiment, self.verbose)


class Evaluator(object):
    ''' An object that encapsulates model evaluation '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.dataloader = dataloader

        self.should_exit = False
        signal.signal(signal.SIGHUP, self.on_training_complete)

        self.observer = None

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.modules = {
            'model': model
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def evaluate(self, batch):
        ''' Runs one evaluation step '''
        with torch.no_grad():
            self.model.eval()
            _, nll = self.model(batch)

            # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
            # will be a tensor of values that must be summed
            nll = nll.sum()

            # need to use .item() which converts to Python scalar
            # because as a Tensor it accumulates gradients
            return nll.item(), torch.sum(batch['target_lens']).item()


    def evaluate_epoch(self, epoch, experiment, verbose=0):
        ''' Evaluate a single epoch '''
        neg_log_likelihood = metrics.Metric('nll', metrics.format_float)

        def get_description():
            mode_name = 'Test' if self.dataset.split == 'test' else 'Validate'
            description = f'{mode_name} #{epoch}'
            if verbose > 0:
                description += f' {neg_log_likelihood}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        batches = tqdm(
            self.dataloader,
            unit='batch',
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )
        with tqdm_wrap_stdout():
            for batch in batches:
                # run the data through the model
                batches.set_description_str(get_description())
                nll, length = self.evaluate(batch)
                if length:
                    neg_log_likelihood.update(nll / length)

        experiment.log_metric('nll', neg_log_likelihood.average)
        return neg_log_likelihood.average

    def on_new_checkpoint(self, path, experiment, verbose=0):
        ''' Upon receiving a new checkpoint path '''
        epoch, step = restore(
            path,
            self.modules,
            num_checkpoints=self.config.average_checkpoints,
            map_location=self.device.type
        )
        experiment.set_step(step)
        self.evaluate_epoch(epoch, experiment, verbose)

    def on_training_complete(self, signum, frame): # pylint:disable=unused-argument
        ''' Received a SIGHUP indicating the training session has ended '''
        self.should_exit = True

    def shutdown(self):
        ''' Shutdown the current observer '''
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def watch(self, experiment, verbose=0):
        ''' Watch for a new checkpoint and run an evaluation step '''
        # Use a polling observer because slurm doesn't seem to correctly handle inotify events :/
        self.observer = PollingObserver() if self.config.polling else Observer()
        event_handler = CheckpointEventHandler(self.on_new_checkpoint, experiment, verbose)
        self.observer.schedule(event_handler, path=self.config.watch_directory)
        self.observer.start()

        while not self.should_exit:
            time.sleep(1)

        atexit.register(self.shutdown)

    def __call__(self, epoch, experiment, verbose=0):
        ''' Validate the model and store off the stats '''
        enter_mode = experiment.validate
        if self.dataset.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            if self.config.watch_directory:
                self.watch(experiment, verbose)
            else:
                return self.evaluate_epoch(epoch, experiment, verbose)
