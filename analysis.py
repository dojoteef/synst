'''
A module for analyzing the syntax of the model output.
'''
import os
import argparse
from collections import Counter
from contextlib import ExitStack
from itertools import chain

from data import preprocess
from utils.tree import ParseTree


def parse_args(argv=None):
    ''' Parse the arguments '''
    parser = argparse.ArgumentParser(
        description='Analyze parse outputs'
    )
    parser.add_argument(
        '-b',
        '--bpe-path',
        type=str,
        required=True,
        help='The path to the learned bpe'
    )
    parser.add_argument(
        '-s',
        '--span',
        type=int,
        default=6,
        help='What span to use for parse segmentation'
    )
    parser.add_argument(
        '--preprocess-buffer-size',
        type=int,
        default=1000,
        help='Number of lines to preprocess at once'
    )
    parser.add_argument(
        'reference',
        type=str,
        help='The reference text'
    )
    parser.add_argument(
        'prediction',
        type=str,
        help='The predicted text'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=0,
        action='count',
        help='Increase the verbosity level'
    )

    return parser.parse_args(args=argv)


def compute_f1(reference, prediction):
    ''' Compute an F1 score using a bag of words approach to the sequence '''
    matches = Counter(prediction) & Counter(reference)
    match_count = sum(matches.values())
    if match_count:
        recall = match_count / len(reference)
        precision = match_count / len(prediction)
        return (2 * recall * precision) / (recall + precision)
    else:
        return 0


def expand_constituents(line, segment=None):
    '''
    Expand the constituents in the parse, i.e convert constituent chunks
    from <NP3> to NP NP NP.
    '''
    if segment:
        constituents, _ = segment(line)
    else:
        constituents = line.strip().split()

    matches = [ParseTree.CONSTITUENT_REGEX.match(c) for c in constituents]
    return list(chain.from_iterable([[m[1]] * int(m[2]) for m in matches if m]))


def process_file(path, buffer=1000):
    ''' Process the given path return the path for the resultant file '''
    parse_ext = '.parse'
    if parse_ext not in path and not os.path.exists(f'{path}{parse_ext}'):
        preprocess.parse(path, f'{path}{parse_ext}', buffer)

    if parse_ext not in path:
        path = f'{path}{parse_ext}'

    return path


def main(argv=None):
    ''' Main entry point for analyzing the parse '''
    args = parse_args(argv)
    reference_path = process_file(args.reference, args.preprocess_buffer_size)
    prediction_path = process_file(args.prediction, args.preprocess_buffer_size)

    segmenters = [
        preprocess.ParseSegmenter(args.bpe_path, span, 0)
        for span in range(1, args.span + 1)
    ]
    segment_reference = f'span{args.span}' not in args.reference
    segment_prediction = f'span{args.span}' not in args.prediction

    with ExitStack() as stack:
        count = 0
        matches = 0
        f1_score = 0
        reference_file = stack.enter_context(open(reference_path, 'rt'))
        prediction_file = stack.enter_context(open(prediction_path, 'rt'))
        for reference_line, prediction_line in zip(reference_file, prediction_file):
            best_match = 0
            best_f1_score = 0
            if not segment_reference:
                reference = expand_constituents(reference_line)

            for segmenter in segmenters:
                if segment_reference:
                    reference = expand_constituents(reference_line, segmenter)

                prediction = expand_constituents(
                    prediction_line,
                    segmenter if segment_prediction else None
                )

                this_f1_score = compute_f1(reference, prediction)
                if this_f1_score > best_f1_score:
                    best_match = reference == prediction
                    best_f1_score = this_f1_score

            count += 1
            matches += best_match
            f1_score += best_f1_score

        if not count:
            print('No lines of input! Unable to analyze {} and {}.')
            exit(1)

        print(f'F1={f1_score / count:.2%}, Accuracy={matches / count:.2%}')


if __name__ == '__main__':
    main()
