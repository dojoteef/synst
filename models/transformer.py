'''
A module which implements the basic Transformer
'''
import uuid
import threading

import torch
from torch import nn

from models.attention import MultiHeadedAttention
from models.embeddings import PositionEmbedding, TokenEmbedding
from models.utils import LabelSmoothingLoss, Translator
from utils import left_shift, right_shift, triu


class TransformerSublayer(nn.Module):
    '''
    Implements a sub layer of the transformer model, which consists of:
    1) A sub layer module
    2) Followed by dropout
    3) Plus a residual connection
    4) With layer normalization
    '''
    def __init__(self, sublayer, sublayer_shape, dropout_p=0.1):
        ''' Initialize the transformer sublayer '''
        super(TransformerSublayer, self).__init__()

        self.sublayer = sublayer
        self.norm = nn.LayerNorm(sublayer_shape)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        self.norm.reset_parameters()

    def forward(self, inputs, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        return self.norm(inputs + self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs)))


class TransformerFFN(nn.Module):
    ''' Implements the Transformer feed-forward network '''
    def __init__(self, embedding_size, hidden_dim):
        super(TransformerFFN, self).__init__()

        self.relu = nn.ReLU()

        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.hidden.weight, gain)
        nn.init.constant_(self.hidden.bias, 0.)

        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.output.weight, gain)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        return self.output(self.relu(self.hidden(inputs)))


class TransformerEncoderLayer(nn.Module):
    ''' Implements a single encoder layer in a transformer encoder stack '''
    def __init__(self, num_heads, dim, hidden_dim, dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerEncoderLayer, self).__init__()

        self.ffn = TransformerSublayer(
            TransformerFFN(dim, hidden_dim),
            dim, dropout_p
        )

        self.self_attention = TransformerSublayer(
            MultiHeadedAttention(dim, num_heads),
            dim, dropout_p
        )

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']
        state = self.self_attention(
            state, # residual
            state, state, state, mask # passed to multiheaded attention
        )

        state = self.ffn(
            state, # residual
            state # passed to feed-forward network
        )

        return {'state': state, 'mask': mask}


class TransformerDecoderLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, num_heads, dim, hidden_dim, causal=True, span=1, dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerDecoderLayer, self).__init__()

        self.span = span
        self.causal = causal
        self.uuid = uuid.uuid4()

        self.ffn = TransformerSublayer(
            TransformerFFN(dim, hidden_dim),
            dim, dropout_p
        )

        self.self_attention = TransformerSublayer(
            MultiHeadedAttention(dim, num_heads),
            dim, dropout_p
        )

        self.source_attention = TransformerSublayer(
            MultiHeadedAttention(dim, num_heads),
            dim, dropout_p
        )

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()
        self.source_attention.reset_parameters()

    def forward(self, inputs, sources): # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']
        cache = inputs.get('cache')

        kwargs = {}
        if self.causal and cache is not None:
            # If caching, only want the last k=span sequence values. Requires no causal masking.
            residual = state[:, -self.span:]
            kwargs['num_queries'] = self.span
        else:
            # If not caching, use the full sequence and ensure an appropriate causal mask
            residual = state
            kwargs['key_mask'] = mask
            kwargs['attention_mask'] = self.mask(state)

        state = self.self_attention(
            residual, # residual
            state, state, state, **kwargs # passed to multiheaded attention
        )

        source = sources['state']
        kwargs = {'key_mask': sources['mask']}
        if self.causal and cache is not None:
            kwargs['num_queries'] = self.span

        state = self.source_attention(
            state, # residual
            source, source, state, **kwargs # passed to multiheaded attention
        )

        state = self.ffn(
            state, # residual
            state # passed to feed-forward network
        )

        if self.causal and cache is not None:
            cached = cache.get(self.uuid)
            if cached is None:
                cache[self.uuid] = state
            else:
                state = cache[self.uuid] = torch.cat((cached, state), 1)

        return {'state': state, 'mask': mask, 'cache': cache}

    _masks = threading.local()
    def mask(self, inputs):
        '''
        Get a self-attention mask

        The mask will be of shape [T x T] containing elements from the set {0, -inf}

        Input shape:  (B x T x E)
        Output shape: (T x T)
        '''
        if not self.causal:
            return None

        dim = inputs.shape[1]
        device = inputs.device
        mask_store = TransformerDecoderLayer._masks.__dict__
        if device not in mask_store:
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, self.span, self.span)

        mask = mask_store[device]
        if mask.shape[0] < dim:
            mask = mask.resize_(dim, dim).fill_(float('-inf'))
            mask_store[device] = triu(mask, 1, self.span, self.span)
            mask = mask_store[device]

        return mask[None, :dim, :dim]


class Transformer(nn.Module):
    ''' The Transformer module '''
    def __init__(self, config, dataset):
        ''' Initialize the Transformer '''
        super(Transformer, self).__init__()

        self.dataset = dataset
        self.span = config.span
        self.embedding = TokenEmbedding(
            dataset.vocab_size,
            config.embedding_size,
            padding_idx=self.padding_idx
        )
        self.position_embedding = PositionEmbedding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        # Allow for overriding the encoders and decoders in dervied classes
        self.encoders = type(self).create_encoders(config)
        self.decoders = type(self).create_decoders(config)

        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )

    @classmethod
    def create_encoders(cls, config):
        ''' Create the transformer encoders '''
        kwargs = {'dropout_p': config.dropout_p}
        args = [config.num_heads, config.embedding_size, config.hidden_dim]
        return nn.ModuleList([
            TransformerEncoderLayer(*args, **kwargs)
            for _ in range(config.num_layers)
        ])

    @classmethod
    def create_decoders(cls, config):
        ''' Create the transformer decoders '''
        kwargs = {'dropout_p': config.dropout_p, 'span': config.span}
        args = [config.num_heads, config.embedding_size, config.hidden_dim]
        return nn.ModuleList([
            TransformerDecoderLayer(*args, **kwargs)
            for _ in range(config.num_layers)
        ])

    @property
    def sos_idx(self):
        ''' Return the sos index '''
        return self.dataset.sos_idx

    @property
    def padding_idx(self):
        ''' Return the padding index '''
        return self.dataset.padding_idx

    def translator(self, config):
        ''' Get a translator for this model '''
        return Translator(config, self, self.dataset)

    def reset_named_parameters(self, modules):
        ''' Get a translator for this model '''
        if 'encoder' in modules:
            for encoder in self.encoders:
                encoder.reset_parameters()
        if 'decoder' in modules:
            for decoder in self.decoders:
                decoder.reset_parameters()
        if 'embeddings' in modules:
            self.embedding.reset_parameters()

    def forward(self, batch): # pylint:disable=arguments-differ
        ''' A batch of inputs and targets '''
        decoded = self.decode(
            self.encode(batch['inputs']),
            right_shift(right_shift(batch['targets']), shift=self.span - 1, fill=self.sos_idx),
        )

        logits = decoded['logits']
        dims = list(range(1, logits.dim()))
        targets = left_shift(batch['targets'])
        nll = self.cross_entropy(logits, targets).sum(dims[:-1])
        smoothed_nll = self.label_smoothing(logits, targets).sum(dims)

        return smoothed_nll, nll

    def encode(self, inputs):
        ''' Encode the inputs '''
        encoded = {
            'state': self.embed(inputs, self.embedding),
            'mask': inputs.eq(self.padding_idx)
        }
        for encoder in self.encoders:
            encoded = encoder(encoded)

        return encoded

    def decode(self, encoded, targets, decoders=None, embedding=None, cache=None, mask=None):
        ''' Decode the encoded sequence to the targets '''
        if decoders is None:
            decoders = self.decoders

        if embedding is None:
            embedding = self.embedding

        decoded = {
            'cache': cache,
            'state': self.embed(targets, embedding),
            'mask': targets.eq(self.padding_idx) if mask is None else mask
        }
        for decoder in decoders:
            decoded = decoder(decoded, encoded)

        # compute projection to the vocabulary
        state = decoded['state']
        if cache is not None:
            state = state[:, -self.span:]

        return {
            'cache': decoded.get('cache'),
            'logits': embedding(state, transpose=True).transpose(2, 1),  # transpose to B x C x ...
        }

    def embed(self, inputs, token_embedding):
        ''' Embed the given inputs '''
        return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))
