'''
Data pre-processing for text files.
'''
import os
import sys
import math
import codecs
import subprocess
import tempfile
from collections import Counter
from multiprocessing import Pool
from contextlib import ExitStack

# pylint:disable=no-name-in-module
from preshed.counter import PreshCounter
# pylint:enable=no-name-in-module
from subword_nmt.learn_bpe import learn_bpe as _learn_bpe
from subword_nmt.apply_bpe import BPE as _BPE
from tqdm import tqdm
import pkg_resources
import spacy
from spacy.lang.char_classes import ALPHA, HYPHENS

import utils.file as file_utils
from utils import tqdm_wrap_stdout
from utils.tree import ParseTree


def _tokenize(language, path, output_path):
    ''' Create a tokenizer function '''
    counter = PreshCounter()
    defaults = spacy.blank(language).Defaults

    rules = defaults.tokenizer_exceptions
    token_match = defaults.token_match
    prefix_search = (spacy.util.compile_prefix_regex(defaults.prefixes).search
                     if defaults.prefixes else None)
    suffix_search = (spacy.util.compile_suffix_regex(defaults.suffixes).search
                     if defaults.suffixes else None)

    # Correct for the fact that spacy does not preserve infix hyphens by default
    punct = r'?";:=,.'
    ignore_infix = r'(?<=[{a}])[{p}]*(?:{h})(?=[{a}])'.format(a=ALPHA, h=HYPHENS, p=punct)

    # Correctly support ending sentences: 1) Start of token followed by a punct, 2)
    # punct followed by punct, 3) two consecutive non-punct chars followed by a
    # punct char, 4) Start of token followed by non-punct char, followed by punct
    punct_infix = r'^[{p}]|(?<=[{p}])[{p}]|(?<=[^{p}]{{2}})[{p}]|(?<=^[^{p}])[{p}]$'.format(p=punct)
    infixes = (
        [punct_infix] +
        [infix for infix in defaults.infixes if infix != ignore_infix]
    )
    infix_finditer = spacy.util.compile_infix_regex(infixes).finditer

    tokenizer = spacy.tokenizer.Tokenizer(
        defaults.create_vocab(), rules=rules,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=token_match
    )

    with ExitStack() as stack:
        input_file = stack.enter_context(open(path, 'rt'))
        output_file = stack.enter_context(open(output_path, 'wt'))
        for text in tokenizer.pipe(input_file):
            tokenized = ' '.join([token.text for token in text if not token.is_space])
            text.count_by(spacy.attrs.ORTH, counts=counter)
            output_file.write(tokenized + '\n')

    return Counter({
        tokenizer.vocab[i].text: c
        for i, c in counter
        if not tokenizer.vocab[i].is_space
    })


def tokenize(path, output_path, buffer=1000):
    ''' Parse a file asynchronously '''
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        results = []
        pool = Pool()
        word_counts = Counter()
        basename = os.path.basename(output_path)
        language = os.path.splitext(basename)[1][1:]
        file_chunks = file_utils.split(path, os.path.join(tmpdir, ''), buffer)
        for chunk in sorted(file_chunks):
            output_chunk = f'{chunk}{basename}'
            results.append(pool.apply_async(_tokenize, [language, chunk, output_chunk]))
            paths.append(output_chunk)
        pool.close()

        results = tqdm(
            results,
            unit='chunk',
            dynamic_ncols=True,
            desc=f'Tokenizing {basename}',
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )
        with tqdm_wrap_stdout():
            for result in results:
                word_counts += result.get()

            pool.join()

        file_utils.join(paths, output_path)
        return word_counts


def learn_bpe(bpe_path, word_counts):
    ''' Learn a BPE '''
    incomplete_path = f'{bpe_path}.incomplete'
    file_utils.try_remove(incomplete_path)

    with codecs.open(incomplete_path, 'w', encoding='utf-8') as bpe_file:
        bpe_dict = [u'{} {}'.format(word, count) for word, count in word_counts]
        _learn_bpe(bpe_dict, bpe_file, 32000, is_dict=True)
    os.rename(incomplete_path, bpe_path)


def _apply_bpe(bpe_path, path, output_path):
    ''' Apply the bpe '''
    with ExitStack() as stack:
        input_file = stack.enter_context(open(path, 'rt'))
        output_file = stack.enter_context(open(output_path, 'wt'))
        bpe_file = stack.enter_context(codecs.open(bpe_path, encoding='utf-8'))

        vocab = set()
        bpe = _BPE(bpe_file)
        for line in input_file:
            leading_whitespace = line[:len(line) - len(line.lstrip('\r\n '))]
            trailing_whitespace = line[len(line.rstrip('\r\n ')) - len(line):]
            subwords = bpe.segment_tokens(line.strip('\r\n ').split(' '))

            vocab.update(subwords)
            output_file.write(leading_whitespace + ' '.join(subwords) + trailing_whitespace)
        return vocab


def apply_bpe(bpe_path, path, output_path, buffer=1000):
    ''' Parse a file asynchronously '''
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        results = []
        pool = Pool()
        vocab = set()
        basename = os.path.basename(output_path)
        file_chunks = file_utils.split(path, os.path.join(tmpdir, ''), buffer)
        for chunk in sorted(file_chunks):
            output_chunk = f'{chunk}{basename}'
            results.append(pool.apply_async(_apply_bpe, [bpe_path, chunk, output_chunk]))
            paths.append(output_chunk)
        pool.close()

        results = tqdm(
            results,
            unit='chunk',
            dynamic_ncols=True,
            desc=f'BPE encoding {basename}',
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )
        with tqdm_wrap_stdout():
            for result in results:
                vocab.update(result.get())

            pool.join()

        file_utils.join(paths, output_path)
        return vocab


def _parse(path):
    ''' Parse the file '''
    parse_script = pkg_resources.resource_filename('__main__', 'scripts/parse_wrapper.sh')
    subprocess.check_call([parse_script, path], stderr=subprocess.PIPE)


def parse(path, output_path, buffer=1000):
    ''' Parse a file asynchronously '''
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        results = []
        pool = Pool()
        basename = os.path.basename(path)
        file_chunks = file_utils.split(path, os.path.join(tmpdir, ''), buffer)
        for chunk in sorted(file_chunks):
            renamed_chunk = f'{chunk}{basename}'

            os.rename(chunk, renamed_chunk)
            results.append(pool.apply_async(_parse, [renamed_chunk]))
            paths.append(renamed_chunk.replace('bpe.32000.', '') + '.parse')
        pool.close()

        results = tqdm(
            results,
            unit='chunk',
            dynamic_ncols=True,
            desc=f'Parsing {basename}',
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )
        with tqdm_wrap_stdout():
            for result in results:
                result.get()

            pool.join()

        file_utils.join(paths, output_path)


class ParseSegmenter(object):
    ''' An object to generate a shallow chunking of a parse tree '''
    def __init__(self, bpe_path, span, max_span=10, constrain_chunks=False):
        ''' Initialize the chunker '''
        self.span = span
        self.max_span = max_span
        self.constrain_chunks = constrain_chunks

        with open(bpe_path, 'rt') as bpe_file:
            self.bpe = _BPE(bpe_file)

    @property
    def node_splitter(self):
        ''' Get the bpe based node splitter '''
        return lambda token: self.bpe.segment(token).split()

    def desired_parse_leaves(self, tree):
        ''' Get the desired number of parse leaves given the tree '''
        return (
            min(max(math.ceil(math.sqrt(len(tree.leaves))), 1), self.span)
            if self.constrain_chunks else self.span
        )

    def __call__(self, line):
        ''' Function to generate a shallow chunking of a parse tree '''
        spans = []
        constituents = []

        tree = ParseTree.from_string(line.strip(), self.node_splitter)
        segments = tree.segment(self.desired_parse_leaves(tree))

        for token in segments:
            match = ParseTree.CONSTITUENT_REGEX.match(token)
            if match:
                span = int(match[2])
                if self.max_span and span > self.max_span:
                    return None, None

                span_mod = span % self.span
                if span > self.span:
                    # Need to break this down into sub-spans because it's greater than
                    # desired span. This happens if a single BPE encoded word has more
                    # tokens than the desired span. Things like URLs and foreign words can
                    # trigger this behavior.
                    count = (span // self.span)
                    constituents.append(f'<{match[1]}{self.span}>')
                    constituents.extend([f'<CONT{self.span}>'] * (count - 1))

                    spans.extend([self.span] * count)
                    if span_mod:
                        spans.append(span_mod)
                        constituents.append(f'<CONT{span_mod}>')
                else:
                    spans.append(span)
                    constituents.append(match[0])

        return constituents, spans


def _get_parse_vocab(path, segmenters):
    ''' Create a tokenizer function '''
    vocab = set()

    with open(path, 'rt') as file:
        for line in file:
            for segmenter in segmenters:
                constituents, _ = segmenter(line)
                vocab.update(constituents)
        return vocab


def get_parse_vocab(path, segmenters, buffer=1000):
    ''' Parse a file asynchronously '''
    with tempfile.TemporaryDirectory() as tmpdir:
        results = []
        pool = Pool()
        vocab = set()
        basename = os.path.basename(path)
        file_chunks = file_utils.split(path, os.path.join(tmpdir, ''), buffer)
        for chunk in sorted(file_chunks):
            results.append(pool.apply_async(_get_parse_vocab, [chunk, segmenters]))
        pool.close()

        results = tqdm(
            results,
            unit='chunk',
            dynamic_ncols=True,
            desc=f'Extracting vocab: {basename}',
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )
        with tqdm_wrap_stdout():
            for result in results:
                vocab.update(result.get())

            pool.join()

        return vocab
