from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
VERSION = __version__

import os


BATCH_SIZE = os.getenv('BATCH_SIZE', 256)
MODEL = os.getenv('MODEL', 'model.h5')
WEIGHTS = os.getenv('WEIGHTS', 'weights.h5')
WORD_DICT = os.getenv('WORD_DICT', 'word-dict.json')

# this is specific to the model
MAX_STRING_LEN = os.getenv('MAX_STRING_LEN', 255)


import sys

if os.path.exists(os.path.join(sys.prefix, 'csirtg_domainsml_tf', 'data', MODEL)):
    MODEL = os.path.join(sys.prefix, 'csirtg_domainsml_tf', 'data', MODEL)
    WEIGHTS = os.path.join(sys.prefix, 'csirtg_domainsml_tf', 'data', WEIGHTS)
    WORD_DICT = os.path.join(sys.prefix, 'csirtg_domainsml_tf', 'data', WORD_DICT)

elif os.path.exists(os.path.join('/usr', 'local',  'csirtg_domainsml_tf', 'data', MODEL)):
    MODEL = os.path.join('/usr', 'local',  'csirtg_domainsml_tf', 'data', MODEL)
    WEIGHTS = os.path.join('/usr', 'local', 'csirtg_domainsml_tf', 'data', WEIGHTS)
    WORD_DICT = os.path.join('/usr', 'local', 'csirtg_domainsml_tf', 'data', WORD_DICT)

else:
    MODEL = os.path.join('%s/../data/%s' % (os.path.dirname(__file__), MODEL))
    WEIGHTS = os.path.join('%s/../data/%s' % (os.path.dirname(__file__), WEIGHTS))
    WORD_DICT = os.path.join('%s/../data/%s' % (os.path.dirname(__file__), WORD_DICT))
