'''
A module to handle metrics
'''
import copy
import os
import shutil
import numbers
import tempfile
from functools import partial

import numpy as np
import torch

import utils


def format_time(seconds):
    ''' Format time in h:mm:ss.ss format '''
    hour = 60 * 60
    hours = int(seconds // hour)
    minutes = int((seconds % hour) // 60)
    return f'{hours}:{minutes:>02}:{seconds % 60.:>05.2f}'


def format_basic(value, format_spec=''):
    ''' Wrapper around format() for use in functools.partial '''
    return format(value, format_spec)

# pylint:disable=invalid-name
format_int = partial(format_basic, format_spec='.0f')
format_float = partial(format_basic, format_spec='.2f')
format_scientific = partial(format_basic, format_spec='.4g')
format_percent = partial(format_basic, format_spec='.2%')
# pylint:enable=invalid-name


class Metric(object):
    ''' Class that represents a metric '''
    def __init__(self, name, formatter=format, default_format_str='g(a)', max_history=None):
        super(Metric, self).__init__()

        self.name = name
        self.formatter = formatter
        self.default_format_str = default_format_str
        self.max_history = max_history

        self.counts, self.values, self.min, self.max = self.reset()

    def reset(self):
        ''' Reset the metrics '''
        self.counts = []
        self.values = []
        self.min = float('inf')
        self.max = float('-inf')
        return self.counts, self.values, self.min, self.max

    def update(self, value, count=1):
        ''' Update the value and counts '''
        self.counts.append(count)
        self.values.append(value)

        average = value / count
        self.min = min(self.min, average)
        self.max = max(self.max, average)

        if self.max_history and len(self.counts) > self.max_history:
            self.counts = self.counts[1:]
            self.values = self.values[1:]

    def updates(self, values, counts=1):
        ''' Update multiple values at once '''
        if isinstance(counts, numbers.Number):
            counts = [counts] * len(values)

        self.counts.extend(counts)
        self.values.extend(values)
        if self.max_history:
            # pylint thinks self.max_history is None...
            # pylint:disable=invalid-unary-operand-type
            self.counts = self.counts[-self.max_history:]
            self.values = self.values[-self.max_history:]
            # pylint:enable=invalid-unary-operand-type

        averages = [value / count for count, value in zip(counts, values)]
        self.min = min(self.min, min(averages))
        self.max = max(self.max, max(averages))

    @property
    def last_count(self):
        ''' Return the last recorded count of the metric'''
        # fancy way to return the last count or zero
        return len(self.counts) and self.counts[-1]

    @property
    def last_value(self):
        ''' Return the last recorded value of the metric '''
        # fancy way to return the last value or zero
        return len(self.values) and self.values[-1]

    @property
    def last_average(self):
        ''' Return the last recorded value of the metric '''
        # fancy way to return the last value or zero
        return self.last_value / max(self.last_count, 1)

    @property
    def total(self):
        ''' Return the current total '''
        return sum(self.values)

    @property
    def total_count(self):
        ''' Return the current total count '''
        return sum(self.counts)

    @property
    def average(self):
        ''' Return the current average value '''
        return self.total / max(self.total_count, 1)

    @property
    def var(self):
        ''' Return the variance of the values '''
        # Need to use a weighted average since each value has an associated count
        counts = np.array(self.counts)
        values = np.array(self.values)
        weights = counts / self.total_count
        return np.average((values - self.average) ** 2, weights=weights)

    @property
    def std(self):
        ''' Return the standard deviation of the values '''
        return np.sqrt(self.var)

    def __format__(self, format_str):
        ''' Return a formatted version of the metric '''
        formatted = f'{self.name}='
        format_str = format_str or self.default_format_str

        compact = True
        paren_depth = 0
        for format_spec, next_format_spec in utils.pairwise(format_str, True):
            if format_spec == 'l':
                compact = False
            elif format_spec == 'c':
                compact = True
            elif format_spec == '(':
                formatted += '('
                paren_depth += 1
            elif format_spec == ')':
                formatted += ')'
                paren_depth -= 1
            elif format_spec == 'C':
                if not compact:
                    formatted += f'last_count='
                formatted += f'{self.formatter(self.last_count)}'
            elif format_spec == 'V':
                if not compact:
                    formatted += f'last_value='
                formatted += f'{self.formatter(self.last_value)}'
            elif format_spec == 'g':
                if not compact:
                    formatted += f'last_avg='
                formatted += f'{self.formatter(self.last_average)}'
            elif format_spec == 'a':
                if not compact:
                    formatted += f'avg='
                formatted += f'{self.formatter(self.average)}'
            elif format_spec == 't':
                if not compact:
                    formatted += f'total='
                formatted += f'{self.formatter(self.total)}'
            elif format_spec == 'm':
                if not compact:
                    formatted += f'min='
                formatted += f'{self.formatter(self.min)}'
            elif format_spec == 'x':
                if not compact:
                    formatted += f'max='
                formatted += f'{self.formatter(self.max)}'
            elif format_spec == 's':
                if not compact:
                    formatted += f'std='
                formatted += f'{self.formatter(self.std)}'
            elif format_spec == 'v':
                if not compact:
                    formatted += f'var='
                formatted += f'{self.formatter(self.var)}'
            else:
                raise ValueError(f'Unknown format specifier {format_spec}')

            if paren_depth and format_spec != '(' and next_format_spec != ')':
                formatted += ','
                if not compact:
                    formatted += ' '

        return formatted

    def __str__(self):
        ''' Return a string representation of the metric '''
        return self.__format__(self.default_format_str)


class MetricStore(object):
    ''' A collection of metrics '''
    def __init__(self, path=None, default_format_str='c'):
        super(MetricStore, self).__init__()

        self.path = path
        self.metrics = {}
        self.default_format_str = default_format_str

    @property
    def directory(self):
        ''' The directory where the metrics are located '''
        return os.path.dirname(self.path) if self.path else None

    def __getitem__(self, key):
        ''' Return the requested metric '''
        return self.metrics[key]

    def add(self, metrics):
        ''' Adds a copy of the Metrics to the store '''
        if isinstance(metrics, Metric):
            # if you pass a single metric
            self.metrics[metrics.name] = metrics
        else:
            # metrics must otherwise be iterable
            self.metrics.update({metric.name: copy.deepcopy(metric) for metric in metrics})

    def save(self):
        ''' Save the metrics to disk '''
        if not self.path:
            raise RuntimeError('Trying to save to disk, but no path was specified!')

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        torch.save(self, self.path)

        with tempfile.NamedTemporaryFile() as temp_checkpoint_file:
            torch.save(self, temp_checkpoint_file)

            shutil.copy(temp_checkpoint_file.name, f'{self.path}.incomplete')
            os.rename(f'{self.path}.incomplete', self.path)

    def load(self):
        ''' Load the metrics from disk '''
        return torch.load(self.path) if self.path and os.path.isfile(self.path) else self

    def __str__(self):
        ''' Return a string representation of the metric store '''
        return self.__format__(self.default_format_str)

    def __format__(self, format_str):
        ''' Return a formatted version of the metric '''
        format_str = format_str or self.default_format_str

        if format_str == 'l':
            return '\n'.join(str(m) for m in self.metrics.values())
        else:
            return ', '.join(str(m) for m in self.metrics.values())
