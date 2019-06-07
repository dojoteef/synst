'''
Initialize the models module
'''
from models.transformer import Transformer
from models.parse import ParseTransformer

MODELS = {
    'transformer': Transformer,
    'parse_transformer': ParseTransformer
}
