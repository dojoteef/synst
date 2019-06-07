'''
Utilities useful for datasets
'''
import os
from functools import partial
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from data.sampler import SequenceLengthSampler


# See https://github.com/tqdm/tqdm#hooks-and-callbacks
class DownloadProgressBar(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def __init__(self, filename):
        ''' '''
        super(DownloadProgressBar, self).__init__(
            unit='B', unit_scale=True, miniters=1, desc=filename)

    def update_to(self, blocks=1, block_size=1, total_size=None):
        """
        blocks  : int, optional
            Number of blocks transferred so far [default: 1].
        block_size  : int, optional
            Size of each block (in tqdm units) [default: 1].
        total_size  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if total_size:
            self.total = total_size

        self.update(blocks * block_size - self.n)  # will also set self.n = blocks * block_size


def maybe_download(filepath, url):
    ''' Download the requested URL to the requested path if it does not already exist '''
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(filepath):
        return filepath

    if 'drive.google.com' in url:
        return download_from_google_drive(filepath, url)
    else:
        return download_url(filepath, url)


def download_url(filepath, url):
    ''' Downloads the given url to the specified file path. '''
    filename = os.path.basename(filepath)
    with DownloadProgressBar(filename) as progress:
        urlretrieve(url, filepath, reporthook=progress.update_to)

    return filepath


def download_from_google_drive(filepath, url):
    '''
    Downloads a file from Google Drive.

    Apparently Google Drive may issue a warning about scanning for viruses and require confirmation
    to continue the download.
    '''
    confirmation_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirmation_token = value

    if confirmation_token:
        url = url + "&confirm=" + confirmation_token

    response = session.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    block_size = 16 * 1024
    filename = os.path.basename(filepath)
    with open(filepath, "wb") as file:
        with DownloadProgressBar(filename) as progress:
            blocks = iter(
                file.write(block)
                for block in response.iter_content(block_size)
                if block
            )

            for i, block in enumerate(blocks):
                progress.update_to(i, block_size, total_size)

    return filepath


def get_dataloader(config, worker_init_fn=None, pin_memory=True, num_devices=1, shuffle=False):
    ''' Utility function that gets a data loader '''
    dataset = config.dataset(config, split=config.split).load()
    if config.batch_method == 'token':
        # Calculate batch sizes for each device. Potentially reduce the batch size on device 0 as
        # the optimization step (all the gradients from all devices) happens on device 0.
        batch_sizes = [config.batch_size - config.batch_size_buffer]
        batch_sizes += [config.batch_size] * (num_devices - 1)
        batch_sampler = SequenceLengthSampler(
            batch_sizes,
            [(len(d['input']), len(d['target'])) for d in dataset.data],
            shuffle=shuffle,
            granularity=config.token_bucket_granularity
        )
    elif config.batch_method == 'example':
        sampler_fn = RandomSampler if shuffle else SequentialSampler
        batch_sampler = BatchSampler(
            sampler_fn(dataset),
            config.batch_size,
            False
        )
    else:
        raise ValueError('Unknown batch method!')

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(dataset.collate, sort=True),
        num_workers=num_devices,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )
