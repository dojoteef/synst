'''
Data loading for the WMT'14 EN-DE dataset as preprocessed by Google Brain and the WMT'14 EN-FR
dataset.
'''
import copy

from data.annotated import AnnotatedTextDataset


class WMTEnDeDataset(AnnotatedTextDataset):
    ''' Class that encapsulates the WMT En-De dataset '''
    NAME = 'wmt'
    LANGUAGE_PAIR = ('en', 'de')

    URLS = [
        (
            'wmt_en_de.tar.gz',
            'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'
        )
    ]
    SPLITS = {
        'dev': 'newstest2013.tok',
        'valid': 'newstest2013.tok',
        'test': 'newstest2014.tok',
        'train': 'train.tok.clean'
    }


class WMTEnFrDataset(AnnotatedTextDataset):
    ''' Class that encapsulates the WMT En-Fr dataset '''
    NAME = 'wmt'
    LANGUAGE_PAIR = ('en', 'fr')

    URLS = [
        ('dev.tgz', 'http://www.statmt.org/wmt14/dev.tgz'),
        ('test-full.tgz', 'http://www.statmt.org/wmt14/test-full.tgz'),
        ('europarl.tgz', 'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz'),
        ('commoncrawl.tgz', 'http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz'),
    ]
    RAW_SPLITS = {
        'train': [
            ('commoncrawl.fr-en.en', 'commoncrawl.fr-en.fr'),
            ('training/europarl-v7.fr-en.en', 'training/europarl-v7.fr-en.fr')
        ],
        'valid': [
            ('dev/newstest2013.en', 'dev/newstest2013.fr')
        ],
        'test': [
            ('test-full/newstest2014-fren-ref.en.sgm', 'test-full/newstest2014-fren-ref.fr.sgm'),
        ]
    }


class WMTEnFrFullDataset(WMTEnFrDataset):
    ''' Class that encapsulates the WMT En-Fr dataset '''
    NAME = 'wmt_full'
    LANGUAGE_PAIR = ('en', 'fr')

    URLS = WMTEnFrDataset.URLS + [
        ('nc.tgz', 'http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz'),
        ('giga.tar', 'http://www.statmt.org/wmt10/training-giga-fren.tar'),
        ('un.tgz', 'http://www.statmt.org/wmt13/training-parallel-un.tgz')
    ]
    RAW_SPLITS = copy.deepcopy(WMTEnFrDataset.RAW_SPLITS)
    RAW_SPLITS['train'].extend([
        ('training/news-commentary-v9.fr-en.en', 'training/news-commentary-v9.fr-en.fr'),
        ('giga-fren.release2.fixed.en.gz', 'giga-fren.release2.fixed.fr.gz'),
        ('un/undoc.2000.fr-en.en', 'un/undoc.2000.fr-en.fr')
    ])
