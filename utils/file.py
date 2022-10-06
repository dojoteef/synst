'''
Various utilities
'''
import io
import os
import glob
import gzip
import math
import tarfile
import zipfile
import subprocess

from utils import ceildiv, grouper

def try_remove(paths):
    ''' Try to remove the given path(s) '''
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def split(path, prefix=None, num_lines=1000, approx_lines=0):
    '''
    Split a file into chunks of each line length.

    If approx_lines are provided, then max sure the suffix length is long enough for the resultant
    number of files.
    '''
    prefix = prefix or f'{path}.chunk.'
    cmd = ['split']
    if approx_lines:
        approx_files = ceildiv(approx_lines, num_lines)
        suffix_len = math.ceil(math.log(approx_files) / math.log(26))
        cmd += ['-a', f'{suffix_len}']
    cmd += ['-l', f'{num_lines}', path, prefix]

    try:
        subprocess.check_call(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise RuntimeError(f'Unable to split {path}')

    return glob.glob(f'{prefix}*')


def join(paths, output_path, batch_size=100):
    ''' Stitch a bunch of chunks into a single file '''
    incomplete_output_path = f'{output_path}.incomplete'
    with open(incomplete_output_path, 'wt') as output_file:
        try:
            # Concatenate a batch of files at a time, in case the file list is too long
            for batch in grouper(paths, batch_size):
                subprocess.check_call(['cat'] + batch, stdout=output_file, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError(f'Unable to join files into {output_path}')
        os.rename(incomplete_output_path, output_path)


class Open(object):
    '''
    A class that acts like function/context manager similar to the builtin open, but supports
    opening as a gzip file seamlessly if needed.
    '''
    def __init__(self, filename, mode='rb'):
        ''' Initialize the file information '''
        self.file = None
        self.mode = mode
        self.filename = filename

    def open(self):
        ''' Open the given file. Use gzip if needed. '''
        if self.file is None:
            if self.mode.startswith(('w', 'a', 'x')):
                # If opening to write, open as a regular file
                file = open(self.filename, self.mode)
            else:
                # When reading, first try as gzipped
                try:
                    file = gzip.open(self.filename, self.mode)
                    if isinstance(file, io.TextIOWrapper):
                        file.buffer.peek(1)  # pylint:disable=no-member
                    else:
                        file.peek(1)
                except OSError:
                    file = open(self.filename, self.mode)

            self.file = file

        return self.file

    def __enter__(self):
        ''' Context manager opens the file on enter '''
        return self.open()

    def __exit__(self, *args):
        '''
        Context manager closes the file on exit.
        Ignores exceptions (so they are raised normally)
        '''
        self.file.close()
        return False

    def __getattr__(self, name):
        ''' Any missing attributes get forwarded to the underlying file '''
        return getattr(self.open(), name)


def extract_all(filename, path):
    '''
    Extracts all the files in the given archive. Seamlessly determines archive type and compression
    '''
    if tarfile.is_tarfile(filename):
        with tarfile.open(filename, 'r') as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, path)
    elif zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename, 'r') as archive:
            archive.extractall(path)
    else:
        raise ValueError('Unknown archive type!')
