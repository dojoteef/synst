'''
Plot some dataset analyses
'''
import argparse
from types import SimpleNamespace

from matplotlib import pyplot as plt

from data import DATASETS


TINY_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24


def parse_args(argv=None):
    ''' Defines the preprocessing specific arguments '''
    parser = argparse.ArgumentParser(
        description='Generate dataset plots'
    )
    parser.add_argument(
        '-D',
        '--datasets',
        type=str,
        action='append',
        choices=[d for d in DATASETS if 'parsed' in d],
        help='Names of the datasets to load'
    )
    parser.add_argument(
        '-d',
        '--data-directories',
        type=str,
        action='append',
        help='Location of the data'
    )
    parser.add_argument(
        '-p',
        '--preprocess-directories',
        type=str,
        action='append',
        help='Location for the preprocessed data'
    )
    parser.add_argument(
        '-s',
        '--spans',
        type=int,
        nargs='*',
        default=[2, 4, 6, 8, 10],
        help='Which spans to plot for the datasets'
    )
    parser.add_argument(
        'figure',
        type=str,
        help='What is the name of the figure to save'
    )

    return parser.parse_args(args=argv)


def main(argv=None):
    ''' The main entry-point to generate our dataset plots '''
    dataset_spans = {}
    args = parse_args(argv)
    for dataset_name, data_directory, preprocess_directory in zip(
            args.datasets, args.data_directories, args.preprocess_directories
    ):
        span_list = []
        for span in sorted(args.spans):
            config = SimpleNamespace()
            config.span = span
            config.data_directory = data_directory
            config.preprocess_directory = preprocess_directory

            config.max_span = 0
            config.max_examples = 0
            config.max_line_length = 0
            config.max_input_length = 0
            config.max_target_length = 0
            config.preprocess_buffer_size = 12500

            dataset = DATASETS[dataset_name](config, split='valid').load()
            span_list.append((span, dataset.stats['Constituent Spans'].average))

        dataset_spans[dataset_name] = span_list

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=TINY_SIZE)     # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Plot dataset statistics
    with plt.style.context('seaborn-colorblind'):
        plt.figure(figsize=(6, 3))
        plt.title('Chunk Size given k')
        for dataset_name, span_list in dataset_spans.items():
            spans, average_spans = zip(*span_list)
            plt.plot(spans, average_spans, label=dataset_name)

        plt.xlabel('k')
        plt.xticks(args.spans)
        plt.ylabel('Average Chunk Size')
        plt.legend(loc='best', bbox_to_anchor=(0.5, 0.01, 0.5, 0.5), borderaxespad=0.)
        plt.savefig(f'{args.figure}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
