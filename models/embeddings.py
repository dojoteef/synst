'''
A module which implements various types of embeddings
'''
import threading

import torch
from torch import nn

from torch.nn import functional as F


class TokenEmbedding(nn.Embedding):
    ''' An embedding layer used for the transformer '''
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
        super(TokenEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        self.scale = embedding_dim ** 0.5
        nn.init.constant_(self.weight[padding_idx], 0)
        nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)

    def forward(self, inputs, transpose=False): # pylint:disable=arguments-differ
        ''' Implement the forward pass of the embedding '''
        if transpose:
            return F.linear(inputs, self.weight)
        else:
            return self.scale * super(TokenEmbedding, self).forward(inputs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        '''
        Not sure if this is the best approach, but override the internal function to support loading
        from a different sized vocabulary.
        '''
        if strict:
            super(TokenEmbedding, self)._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs
            )
        else:
            # Support loading from a different sized vocabulary
            weight_name = prefix + 'weight'
            for key, param in state_dict.items():
                if key == weight_name:
                    old_vocab_size = len(param)
                    new_vocab_size = len(self.weight)
                    vocab_size_diff = new_vocab_size - old_vocab_size
                    if vocab_size_diff > 0:
                        param = torch.cat((param, self.weight[old_vocab_size:]), 0)
                    else:
                        param = param[:new_vocab_size]

                    self.weight.data.copy_(param)


class PositionEmbedding(nn.Module):
    ''' Produce position embeddings '''
    def __init__(self, dim, freq=1e4):
        ''' Initialize the PositionEmbedding '''
        super(PositionEmbedding, self).__init__()

        self.dim = dim
        self.freq = freq

    _embeddings = threading.local()
    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' Implement the forward pass of the embedding '''
        device = inputs.device
        max_length = inputs.shape[1]
        embedding_store = PositionEmbedding._embeddings.__dict__
        device_store = embedding_store.get(device, {})
        if (
                not device_store or
                self.dim not in device_store or
                device_store[self.dim].shape[0] < max_length
        ):
            positions = torch.arange(0., max_length, device=device).unsqueeze(1)

            # the tensor2tensor code is slightly different than described in the paper
            # dividing by (self.dim - 2) produces nearly identical results to their version
            # when comparing the tensorflow results to these torch results
            dims = torch.arange(0., self.dim, 2., device=device).unsqueeze(0) / (self.dim - 2)

            sin = torch.sin(positions / torch.pow(self.freq, dims))
            cos = torch.cos(positions / torch.pow(self.freq, dims))

            embeddings = torch.stack((sin, cos), 0)
            device_store[self.dim] = embeddings.transpose(0, 1).contiguous().view(-1, self.dim)

        embeddings = device_store[self.dim]
        embedding_store[device] = device_store
        return embeddings[:max_length].unsqueeze(0)
