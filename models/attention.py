'''
A module which implements various attention mechanisms
'''
import torch
from torch import nn
from torch.nn import functional as F

from utils import same_tensor


class MultiHeadedAttention(nn.Module):
    ''' Implement a multi-headed attention module '''
    def __init__(self, embed_dim, num_heads=1):
        ''' Initialize the attention module '''
        super(MultiHeadedAttention, self).__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.scale = self.projection_dim ** -0.5

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim
        projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def attention(self, values, keys, queries, key_mask=None, mask=None):
        ''' Scaled dot product attention with optional masks '''
        logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))
        if mask is not None:
            logits += mask

        if key_mask is not None:
            logits_shape = logits.shape
            batch_size = logits_shape[0] // self.num_heads
            logits = logits.view(batch_size, self.num_heads, logits_shape[1], logits_shape[2])
            logits.masked_fill_(key_mask[:, None, None], float('-inf'))
            logits = logits.view(logits_shape)

        attended = torch.bmm(F.softmax(logits, dim=-1), values)

        # By this point the values, keys, and queries all have B * H as their first dimension
        batch_size = queries.shape[0] // self.num_heads
        return attended.view(
            batch_size,
            self.num_heads,
            -1,
            self.projection_dim
        ).transpose(2, 1).contiguous().view(
            batch_size,
            -1,
            self.num_heads * self.projection_dim
        )

    def forward(self, values, keys, queries, # pylint:disable=arguments-differ
                key_mask=None, attention_mask=None, num_queries=0):
        ''' Forward pass of the attention '''
        # pylint:disable=unbalanced-tuple-unpacking
        if same_tensor(values, keys, queries):
            values, keys, queries = self.project(values, chunks=3)
        elif same_tensor(values, keys):
            values, keys = self.project(values, chunks=2)
            queries, = self.project(queries, 2)
        else:
            values, = self.project(values, 0)
            keys, = self.project(keys, 1)
            queries, = self.project(queries, 2)
        # pylint:enable=unbalanced-tuple-unpacking

        if num_queries:
            queries = queries[:, -num_queries:]

        attended = self.attention(values, keys, queries, key_mask, attention_mask)
        return self.output_projection(attended)
