'''
SynST

--
Main entry point for translating from SynST
'''

from __future__ import print_function

import os
import sys
import timeit
from contextlib import ExitStack

import torch
from tqdm import tqdm

from utils import profile
from utils import tqdm_wrap_stdout


class Translator(object):
    ''' An object that encapsulates model evaluation '''
    CURRENT = None

    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.dataloader = dataloader
        self.translator = model.translator(config).to(device)

        self.modules = {
            'model': model
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    @property
    def annotation_sos_idx(self):
        ''' Get the annotation sos index '''
        return self.dataset.sos_idx - self.dataset.reserved_range

    @property
    def annotation_eos_idx(self):
        ''' Get the annotation eos index '''
        return self.dataset.eos_idx - self.dataset.reserved_range

    @property
    def padding_idx(self):
        ''' Get the padding index '''
        return self.dataset.padding_idx

    def translate_all(self, output_file, epoch, experiment, verbose=0):
        ''' Generate all predictions from the dataset '''
        def get_description():
            description = f'Generate #{epoch}'
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
            ordered_outputs = []
            for batch in batches:
                # run the data through the model
                batches.set_description_str(get_description())
                sequences = self.translator.translate(batch)

                if self.config.timed:
                    continue

                target_sequences = next(iter(sequences.values()))
                for i, example_id in enumerate(batch['example_ids']):
                    outputs = []
                    if verbose > 1:
                        trim = verbose < 2
                        join = verbose < 3
                        for key in sequences.keys():
                            sequence = sequences[key][i]
                            sequence = ' '.join(self.dataset.decode(sequence, join, trim))
                            outputs.append(f'{key}: {sequence}\n')
                        outputs.append(f'+++++++++++++++++++++++++++++\n')
                    else:
                        sequence = target_sequences[i]
                        decoded = ' '.join(self.dataset.decode(sequence, trim=not verbose))
                        outputs.append(f'{decoded}\n')

                    if self.config.order_output:
                        ordered_outputs.append((example_id, outputs))
                    else:
                        output_file.writelines(outputs)

            for _, outputs in sorted(ordered_outputs, key=lambda x: x[0]): # pylint:disable=consider-using-enumerate
                output_file.writelines(outputs)

    def __call__(self, epoch, experiment, verbose=0):
        ''' Generate from the model '''
        enter_mode = experiment.validate
        if self.dataset.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            if not os.path.isdir(self.config.output_directory):
                os.makedirs(self.config.output_directory)

            if self.config.timed:
                Translator.CURRENT = self
                stmt = f'Translator.CURRENT.translate_all(None, {epoch}, None, {verbose})'
                timing = timeit.timeit(stmt, stmt, number=self.config.timed, globals=globals())
                print(f'Translation timing={timing/self.config.timed}')
            else:
                step = experiment.curr_step
                output_filename = self.config.output_filename or f'translated_{step}.txt'
                output_path = os.path.join(self.config.output_directory, output_filename)
                output_file = stack.enter_context(open(output_path, 'wt'))

                if verbose:
                    print(f'Outputting to {output_path}')

                self.translate_all(output_file, epoch, experiment, verbose)
