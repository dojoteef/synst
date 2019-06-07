'''
A module implementing various data samplers for datasets.
'''
import heapq

import numpy as np
from torch.utils.data import Sampler

from utils import ceildiv


class TokenBucket(object):
    ''' A bucket of sequence ids '''
    def __init__(self, max_lengths):
        ''' Initialize the bucket '''
        self.max_lengths = max_lengths
        self.reset()

    def reset(self):
        ''' Reset the bucket '''
        self.heap = [(0, i, length, []) for i, length in enumerate(self.max_lengths)]

    def try_add(self, sequence_id, sequence_lengths):
        ''' Try to add the given example '''
        full = []
        while self.heap:
            total_length, device_id, max_length, sequence_ids = heapq.heappop(self.heap)
            if total_length + sequence_lengths > max_length:
                full.append((total_length, device_id, max_length, sequence_ids))
            else:
                sequence_ids.append(sequence_id)
                total_length += sequence_lengths
                heapq.heappush(self.heap, (total_length, device_id, max_length, sequence_ids))
                break

        if self.heap:
            # Add back any full device lists
            while full:
                heapq.heappush(self.heap, full.pop())
        else:
            # All batches were full
            self.reset()
            return self.extract_batch(full)

    def extract_batch(self, iterable):
        ''' Extract a batch from the iterable '''
        _, batch = zip(*sorted(
            (device_id, sequence_ids)
            for  _, device_id, _, sequence_ids in iterable
        ))

        return batch

    def get_batch(self):
        ''' Get the current batch '''
        return self.extract_batch(self.heap)


class SequenceLengthSampler(Sampler):
    ''' A sampler that tries to select batches that have a given total sequence length '''
    def __init__(self, max_lengths, sequence_lengths, shuffle=False, granularity=5):
        '''
        Initializer the sequence length sampler

        Inputs:
        max_lengths - a list of lengths of the desired total sequence length for each device
        lengths - a list containing the length for each example in the dataset
        '''
        super(SequenceLengthSampler, self).__init__(sequence_lengths)

        self.shuffle = shuffle
        self.granularity = granularity
        self.max_lengths = max_lengths
        self.sequence_lengths = sequence_lengths

        # Initial estimate of the number of batches
        self.num_batches = ceildiv(np.sum(sequence_lengths), np.sum(max_lengths))

    def __len__(self):
        ''' Estimate the number of batches per iteration '''
        return self.num_batches

    def __iter__(self):
        ''' Produce batches according the given lengths '''
        buckets = {}
        num_batches = 0
        sequence_lengths = list(enumerate(self.sequence_lengths))
        if self.shuffle:
            np.random.shuffle(sequence_lengths)

        for idx, lengths in sequence_lengths:
            bucket_idx = ceildiv(max(lengths), self.granularity)
            bucket = buckets.get(bucket_idx, None)
            if not bucket:
                bucket = TokenBucket(self.max_lengths)

            batch = bucket.try_add(idx, sum(lengths))
            if batch:
                # Bucket was full so yield a batch
                num_batches += 1
                yield batch

                # Add to the bucket now that it's been emptied
                bucket.try_add(idx, sum(lengths))

            buckets[bucket_idx] = bucket

        # Go through all buckets to see if any can yield a batch
        for bucket in buckets.values():
            batch = bucket.get_batch()
            if all(batch):
                # Bucket had a non-empty batch left
                num_batches += 1
                yield batch

        # Update the batch estimate
        self.num_batches = num_batches
