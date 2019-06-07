'''
Initialize the actions module
'''
from actions.train import Trainer
from actions.evaluate import Evaluator
from actions.translate import Translator

class Pass(object):
    ''' Action that does nothing... '''
    def __init__(self, *args, **kwargs):
        ''' Do nothing '''
        pass

    def __call__(self, *args, **kwargs):
        ''' Do nothing '''
        pass
