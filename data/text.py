'''
Base for text datasets.
'''
import collections
from itertools import chain

import torch
from torch import nn
from torch.utils.data import Dataset

import metrics


PAD = '<PAD>'
START_OF_SUMMARY = '<SOS>'
END_OF_SUMMARY = '<EOS>'


class TextDataset(Dataset):
    ''' Base class for a text dataset '''
    def __init__(self, config, split='train'):
        ''' Initialize the TextDataset '''
        self.data = []
        self.skipped = 0
        self.token2id = {}
        self.id2token = []
        self.split = split
        self.config = config
        self.reserved_range = len(self.id2token)

    def __len__(self):
        ''' Get the length of the dataset '''
        return len(self.data)

    def tensorize(self, index):
        ''' Tensorize the specified example index '''
        return dict((k, torch.LongTensor(v)) for k, v in self.data[index].items())

    def __getitem__(self, index):
        ''' Get the story/stories at the specified index/indices '''
        if isinstance(index, collections.Sequence):
            return tuple(
                (i, self.tensorize(i)) for i in index
            )
        else:
            return (index, self.tensorize(index))

    @property
    def padding_idx(self):
        ''' Return the padding value '''
        return self.token2id[PAD]

    @property
    def sos_idx(self):
        ''' Return the start of summary value '''
        return self.token2id[START_OF_SUMMARY]

    @property
    def eos_idx(self):
        ''' Return the end of summary value '''
        return self.token2id[END_OF_SUMMARY]

    @property
    def special_tokens(self):
        ''' Return the full list of special tokens '''
        return set(range(self.reserved_range, len(self.id2token)))

    @property
    def vocab_size(self):
        ''' Return the vocab size '''
        return len(self.id2token)

    @property
    def annotation_vocab_size(self):
        ''' A derived property of self.vocab_size '''
        return self.vocab_size - self.reserved_range

    @property
    def max_input_length(self):
        ''' Returns the max input length '''
        return self.config.max_input_length

    @property
    def max_target_length(self):
        ''' Returns the max target length '''
        return self.config.max_target_length

    def collate_field(self, batch, field_name, values):
        ''' Collate a specific field '''
        batch[field_name + 's'] = nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=self.padding_idx)
        batch[field_name + '_lens'] = torch.LongTensor([len(sequence) for sequence in values])

    def collate(self, data, sort=False):
        ''' Collate the data into a batch '''
        if not data:
            return []

        def make_batch(ids, examples):
            ''' Make a batch given a dict of lists with inputs and targets '''
            # must store off lengths before padding sequence
            batch = {'example_ids': ids}
            for key, values in examples.items():
                self.collate_field(batch, key, values)

            return batch

        def flatten(examples):
            ''' Flattens a list of dicts into a single dict of lists '''
            ids, examples = zip(*examples)
            keys = examples[0].keys()

            zipped = zip(*(e.values() for e in examples))
            return ids, {k: next(zipped) for k in keys}

        def sorter(examples, key='input'):
            ''' Sort the list of examples based on the length of the sequence for the given key '''
            return sorted(examples, key=lambda x: len(x[1][key]), reverse=True)

        if any(
                isinstance(d, tuple) and len(d) and
                isinstance(d[0], collections.Sequence)
                for d in data
        ):
            if sort:
                # Sort within each chunk
                data = [sorter(d) for d in data]

            ids, examples = zip(*(flatten(d) for d in data))
            ids = chain.from_iterable(ids)
            keys = examples[0].keys()
            examples = {k: list(chain.from_iterable(e[k] for e in examples)) for k in keys}
            batch = make_batch(ids, examples)
            batch['chunk_sizes'] = [len(l) for l in data]
            return batch
        else:
            if sort:
                data = sorter(data)

            return make_batch(*flatten(data))

    def decode(self, text, join=True, trim=True):
        ''' Decode a given string by mapping token ids to corresponding text '''
        word = []
        decoded = []
        for i in text:
            if not trim or i not in self.special_tokens:
                byte_pair = self.id2token[int(i)]
                if join:
                    if byte_pair.endswith('@@'):
                        word.append(byte_pair[:-2])
                    else:
                        word.append(byte_pair)
                        decoded.append(''.join(word))
                        word = []
                else:
                    decoded.append(byte_pair)

            if i == self.eos_idx:
                break

        return decoded

    def load(self, preprocess=True):
        ''' Load the dataset '''
        if preprocess:
            self.preprocess()

        self.load_vocab()
        self.load_text()

        return self

    @property
    def stats(self):
        ''' Return the dataset stats '''
        metric_store = metrics.MetricStore(default_format_str='l')
        examples = metrics.Metric('Examples', metrics.format_int, 'g')
        examples.update(len(self))
        metric_store.add(examples)

        if self.skipped:
            skipped = metrics.Metric('Skipped', metrics.format_percent, 'g')
            skipped.update(self.skipped, self.skipped + len(self))
            metric_store.add(skipped)

        vocab_size = metrics.Metric('Vocab Size', metrics.format_int, 'g')
        vocab_size.update(self.vocab_size)
        metric_store.add(vocab_size)

        input_lengths, target_lengths = zip(*[
            (len(d['input']), len(d['target']))
            for d in self.data
        ])

        input_length = metrics.Metric('Input Length', metrics.format_int, 'l(max)')
        input_length.updates(input_lengths)
        metric_store.add(input_length)

        target_length = metrics.Metric('Target Length', metrics.format_int, 'l(max)')
        target_length.updates(target_lengths)
        metric_store.add(target_length)

        return metric_store

    def preprocess(self):
        ''' Do any data preprocessing including downloading the data if needed '''
        raise NotImplementedError('Subclasses must implement preprocess!')

    def load_text(self):
        ''' Load the text '''
        raise NotImplementedError('Subclasses must implement load_text!')

    def load_vocab(self, preprocessing=False): # pylint:disable=unused-argument
        ''' Load the vocab '''
        self.reserved_range = len(self.id2token)

        self.token2id[PAD] = len(self.id2token)
        self.id2token.append(PAD)

        self.token2id[START_OF_SUMMARY] = len(self.id2token)
        self.id2token.append(START_OF_SUMMARY)

        self.token2id[END_OF_SUMMARY] = len(self.id2token)
        self.id2token.append(END_OF_SUMMARY)

    def add_datum(self, datum):
        ''' Add a single datum '''
        if self.exceeds_limits(datum):
            self.skipped += 1
            return False

        self.data.append(datum)
        return True

    def exceeds_limits(self, datum):
        ''' Determine if the data limits have been exceeded '''
        return ((self.max_input_length and len(datum['input']) > self.max_input_length) or
                (self.max_target_length and len(datum['target']) > self.max_target_length) or
                (self.config.max_examples and len(self) >= self.config.max_examples))
