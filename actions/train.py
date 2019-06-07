'''
SynST

--
Main entry point for training SynST
'''

from __future__ import print_function

import os
import sys
import time
import shutil
from contextlib import ExitStack

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from tqdm import tqdm

import args
import metrics
from actions.evaluate import Evaluator
from data.parallel import chunked_scattering
from models.utils import LinearLRSchedule, WarmupLRSchedule, checkpoint
from utils import profile, tqdm_wrap_stdout, tqdm_unwrap_stdout


class Trainer(object):
    ''' An object that encapsulates model training '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.stopped_early = False
        self.dataloader = dataloader
        self.validation_dataloader = dataloader
        self.last_checkpoint_time = time.time()

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.optimizer = optim.Adam(model.parameters(), config.base_lr, betas=(0.9, 0.98), eps=1e-9)
        if config.lr_scheduler == 'warmup':
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                WarmupLRSchedule(
                    config.warmup_steps
                )
            )
        elif config.lr_scheduler == 'linear':
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                LinearLRSchedule(
                    config.base_lr,
                    config.final_lr,
                    config.max_steps
                )
            )
        elif config.lr_scheduler == 'exponential':
            self.lr_scheduler = ExponentialLR(
                self.optimizer,
                config.lr_decay
            )
        else:
            raise ValueError('Unknown learning rate scheduler!')

        # Initialize the metrics
        metrics_path = os.path.join(self.config.checkpoint_directory, 'train_metrics.pt')
        self.metric_store = metrics.MetricStore(metrics_path)
        self.metric_store.add(metrics.Metric('oom', metrics.format_int, 't'))
        self.metric_store.add(metrics.Metric('nll', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('lr', metrics.format_scientific, 'g', max_history=1))
        self.metric_store.add(metrics.Metric('num_tok', metrics.format_int, 'a', max_history=1000))

        if self.config.early_stopping:
            self.metric_store.add(metrics.Metric('vnll', metrics.format_float, 'g'))

        self.modules = {
            'model': model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def train_epoch(self, epoch, experiment, verbose=0):
        ''' Run one training epoch '''
        oom = self.metric_store['oom']
        learning_rate = self.metric_store['lr']
        num_tokens = self.metric_store['num_tok']
        neg_log_likelihood = self.metric_store['nll']

        def try_optimize(i, last=False):
            # optimize if:
            #  1) last and remainder
            #  2) not last and not remainder
            remainder = bool(i % self.config.accumulate_steps)
            if not last ^ remainder:
                next_lr = self.optimize()

                learning_rate.update(next_lr)
                experiment.log_metric('learning_rate', next_lr)
                return True

            return False

        def get_description():
            description = f'Train #{epoch}'
            if verbose > 0:
                description += f' {self.metric_store}'
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
            i = 1
            nll_per_update = 0.
            length_per_update = 0
            num_tokens_per_update = 0
            for i, batch in enumerate(batches, 1):
                try:
                    nll, length = self.calculate_gradient(batch)
                    did_optimize = try_optimize(i)

                    # record the effective number of tokens
                    num_tokens_per_update += int(sum(batch['input_lens']))
                    num_tokens_per_update += int(sum(batch['target_lens']))

                    if length:
                        # record length and nll
                        nll_per_update += nll
                        length_per_update += length

                    if did_optimize:
                        # advance the experiment step
                        experiment.set_step(experiment.curr_step + 1)

                        num_tokens.update(num_tokens_per_update)
                        neg_log_likelihood.update(nll_per_update / length_per_update)

                        experiment.log_metric('num_tokens', num_tokens_per_update)
                        experiment.log_metric('nll', neg_log_likelihood.last_value)

                        nll_per_update = 0.
                        length_per_update = 0
                        num_tokens_per_update = 0

                except RuntimeError as rte:
                    if 'out of memory' in str(rte):
                        torch.cuda.empty_cache()

                        oom.update(1)
                        experiment.log_metric('oom', oom.total)
                    else:
                        batches.close()
                        raise rte

                if self.should_checkpoint():
                    new_best = False
                    if self.config.early_stopping:
                        with tqdm_unwrap_stdout():
                            new_best = self.evaluate(experiment, epoch, verbose)

                    self.checkpoint(epoch, experiment.curr_step, new_best)

                batches.set_description_str(get_description())
                if self.is_done(experiment, epoch):
                    batches.close()
                    break

            try_optimize(i, last=True)

    def should_checkpoint(self):
        ''' Function which determines if a new checkpoint should be saved '''
        return time.time() - self.last_checkpoint_time > self.config.checkpoint_interval

    def checkpoint(self, epoch, step, best=False):
        ''' Save a checkpoint '''
        checkpoint_path = checkpoint(
            epoch, step, self.modules,
            self.config.checkpoint_directory,
            max_checkpoints=self.config.max_checkpoints
        )

        if best:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
            shutil.copy2(checkpoint_path, best_checkpoint_path)

        self.metric_store.save()
        self.last_checkpoint_time = time.time()

    def evaluate(self, experiment, epoch, verbose=0):
        ''' Evaluate the current model and determine if it is a new best '''
        model = self.modules['model']
        evaluator = Evaluator(args.ArgGroup(None), model, self.validation_dataloader, self.device)
        vnll = evaluator(epoch, experiment, verbose)
        metric = self.metric_store['vnll']
        full_history = metric.values
        metric.update(vnll)
        self.metric_store.save()

        return all(vnll < nll for nll in full_history[:-1])

    def is_done(self, experiment, epoch):
        ''' Has training completed '''
        if self.config.max_steps and experiment.curr_step >= self.config.max_steps:
            return True

        if self.config.max_epochs and epoch >= self.config.max_epochs:
            return True

        if self.config.early_stopping:
            history = self.metric_store['vnll'].values[-self.config.early_stopping - 1:]
            if len(history) == self.config.early_stopping + 1:
                self.stopped_early = all(history[-1] > nll for nll in history[:-1])
                return self.stopped_early

        return False

    def optimize(self):
        ''' Calculate an optimization step '''
        self.lr_scheduler.step()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.lr_scheduler.get_lr()[0]

    def calculate_gradient(self, batch):
        ''' Runs one step of optimization '''
        # run the data through the model
        self.model.train()
        loss, nll = self.model(batch)

        # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
        # will be a tensor of values that must be summed
        nll = nll.sum()
        loss = loss.sum()

        # calculate gradients then run an optimization step
        loss.backward()

        # need to use .item() which converts to Python scalar
        # because as a Tensor it accumulates gradients
        return nll.item(), torch.sum(batch['target_lens']).item()

    def __call__(self, start_epoch, experiment, verbose=0):
        ''' Execute training '''
        with ExitStack() as stack:
            stack.enter_context(chunked_scattering())
            stack.enter_context(experiment.train())

            if start_epoch > 0 or experiment.curr_step > 0:
                # TODO: Hacky approach to decide if the metric store should be loaded. Revisit later
                self.metric_store = self.metric_store.load()

            epoch = start_epoch
            experiment.log_current_epoch(epoch)
            while not self.is_done(experiment, epoch):
                experiment.log_current_epoch(epoch)
                self.train_epoch(epoch, experiment, verbose)
                experiment.log_epoch_end(epoch)
                epoch += 1

            if self.stopped_early:
                print('Stopping early!')
            else:
                new_best = False
                if self.config.early_stopping:
                    new_best = self.evaluate(experiment, epoch, verbose)

                self.checkpoint(epoch, experiment.curr_step, new_best)
