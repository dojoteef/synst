'''
A module which implements a parse based Transformer
'''
from collections import OrderedDict
import torch
from torch import nn

from models.embeddings import TokenEmbedding
from models.transformer import Transformer, TransformerDecoderLayer
from models.utils import ModuleWrapper
from utils import right_shift, left_shift
from utils.beam_search import BeamSearchDecoder


class ParseTransformer(Transformer):
    ''' The ParseTransformer module '''
    def __init__(self, config, dataset):
        ''' Initialize the ParseTransformer '''
        super(ParseTransformer, self).__init__(config, dataset)

        self.span = 1
        args = [config.num_heads, config.embedding_size, config.hidden_dim]
        self.annotation_decoders = nn.ModuleList([
            TransformerDecoderLayer(*args, dropout_p=config.dropout_p)
            for _ in range(config.parse_num_layers)
        ])

        self.annotation_embedding = TokenEmbedding(
            dataset.annotation_vocab_size,
            config.embedding_size,
            padding_idx=self.annotation_padding_idx
        )
        self.annotation_cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.annotation_padding_idx,
            reduction='none'
        )

    @classmethod
    def create_decoders(cls, config):
        ''' Create the transformer decoders '''
        if config.parse_only:
            return None

        kwargs = {'dropout_p': config.dropout_p, 'span': config.span, 'causal': False}
        args = [config.num_heads, config.embedding_size, config.hidden_dim]
        return nn.ModuleList([
            TransformerDecoderLayer(*args, **kwargs)
            for _ in range(config.num_layers)
        ])

    @property
    def reserved_range(self):
        ''' Return the reserved range of annotations '''
        return self.dataset.reserved_range

    @property
    def annotation_sos_idx(self):
        ''' Return the sos index '''
        return self.dataset.sos_idx - self.reserved_range

    @property
    def annotation_padding_idx(self):
        ''' Return the padding index '''
        return self.dataset.padding_idx - self.reserved_range

    def translator(self, config):
        ''' Get a translator for this model '''
        return Translator(config, self, self.dataset)

    def forward(self, batch): # pylint:disable=arguments-differ
        ''' A batch of inputs and targets '''
        encoded = self.encode(batch['inputs'])
        decoded_annotation = self.decode_annotation(
            encoded, right_shift(batch['target_annotations'])
        )

        logits = decoded_annotation['logits']
        loss = nll = self.annotation_cross_entropy(
            logits, left_shift(batch['target_annotations'])
        ).sum(list(range(1, logits.dim() - 1)))

        if self.decoders is not None:
            decoded = self.decode(encoded, batch['masked_targets'])
            logits = decoded['logits']

            nll += self.cross_entropy(
                logits, batch['targets']
            ).sum(list(range(1, logits.dim() - 1)))

            loss += self.label_smoothing(
                logits, batch['targets']
            ).sum(list(range(1, logits.dim())))

        return loss, nll

    def decode_annotation(self, encoded, annotations, cache=None):
        ''' Decode the encoded sequence to the target annotations '''
        return self.decode(
            encoded, annotations,
            self.annotation_decoders,
            self.annotation_embedding,
            mask=annotations.eq(self.annotation_padding_idx),
            cache=cache
        )


class Translator(object):
    ''' An object that encapsulates model evaluation '''
    def __init__(self, config, model, dataset):
        self.config = config
        self.dataset = dataset

        self.encoder = ModuleWrapper(model, 'encode')
        self.decoder = ModuleWrapper(model, 'decode')
        self.annotation_decoder = ModuleWrapper(model, 'decode_annotation')

        self.modules = {
            'model': model
        }

    def to(self, device):
        ''' Move the translator to the specified device '''
        if 'cuda' in device.type:
            self.encoder = nn.DataParallel(self.encoder.cuda())
            self.decoder = nn.DataParallel(self.decoder.cuda())
            self.annotation_decoder = nn.DataParallel(self.annotation_decoder.cuda())

        return self

    @property
    def annotations_only(self):
        ''' Whether to translate annotations only '''
        return self.config.annotations_only or self.modules['model'].decoders is None

    @property
    def reserved_range(self):
        ''' Get the annotation sos index '''
        return self.dataset.reserved_range

    @property
    def annotation_sos_idx(self):
        ''' Get the annotation sos index '''
        return self.dataset.sos_idx - self.reserved_range

    @property
    def annotation_eos_idx(self):
        ''' Get the annotation eos index '''
        return self.dataset.eos_idx - self.reserved_range

    @property
    def padding_idx(self):
        ''' Get the padding index '''
        return self.dataset.padding_idx

    def translate(self, batch):
        ''' Generate with the given batch '''
        with torch.no_grad():
            target_annotations = []
            encoded = self.encoder(batch['inputs'])
            if self.config.gold_annotations:
                for annotation, length in zip(
                        batch['target_annotations'],
                        batch['target_annotation_lens']
                ):
                    # put into proper range
                    target_annotation = annotation[:length] + self.reserved_range

                    target_annotations.append(target_annotation.tolist())
            else:
                if self.config.length_basis:
                    length_basis = batch[self.config.length_basis]
                else:
                    length_basis = [0] * len(batch['inputs'])

                annotation_decoder = BeamSearchDecoder(
                    self.annotation_decoder,
                    self.annotation_eos_idx,
                    self.config,
                    1
                )

                beams = annotation_decoder.initialize_search(
                    [[self.annotation_sos_idx] for _ in range(len(batch['inputs']))],
                    [l + self.config.max_decode_length + 1 for l in length_basis]
                )

                for beam in annotation_decoder.decode(encoded, beams):
                    # put into proper range
                    target_annotation = torch.LongTensor(beam.best_hypothesis.sequence)
                    target_annotation += self.reserved_range

                    target_annotations.append(target_annotation.tolist())

            outputs = OrderedDict()
            if not self.annotations_only:
                target_lengths = []
                masked_targets = []
                for annotation in target_annotations:
                    # strip off start/end tokens
                    annotation = annotation[1:-1]
                    spans = self.dataset.annotation_spans(annotation)
                    masked_target = self.dataset.masked_target(annotation, spans)

                    target_lengths.append(sum(spans) + len(spans))
                    masked_targets.append(torch.LongTensor(masked_target))

                if max(target_lengths) > 0:
                    masked_targets = nn.utils.rnn.pad_sequence(
                        masked_targets, batch_first=True, padding_value=self.padding_idx
                    )

                    targets = []
                    result = self.decoder(encoded, masked_targets)
                    translated = result['logits'].argmax(1).tolist()
                    for length, target in zip(target_lengths, translated):
                        targets.append(target[:length])
                    outputs['targets'] = targets
                else:
                    outputs['targets'] = [[]] * len(masked_targets)

                gold_targets = []
                gold_target_lens = batch['target_lens']
                for i, target in enumerate(batch['targets']):
                    target_len = gold_target_lens[i]
                    gold_targets.append(target[:target_len].tolist())

                outputs['gold_targets'] = gold_targets

            gold_annotations = []
            gold_annotation_lens = batch['target_annotation_lens']
            for i, annotation in enumerate(batch['target_annotations']):
                annotation_len = gold_annotation_lens[i]
                annotation = annotation[:annotation_len] + self.reserved_range
                gold_annotations.append(annotation.tolist())

            outputs['annotations'] = target_annotations
            outputs['gold_annotations'] = gold_annotations
            return outputs
