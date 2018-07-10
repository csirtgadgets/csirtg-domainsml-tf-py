#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import sys, gc
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import textwrap

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from pprint import pprint

from csirtg_domainsml_tf.constants import MODEL, WEIGHTS, MAX_STRING_LEN, WORD_DICT


def normalize_domains(indicators):
    if not isinstance(indicators, list):
        indicators = [indicators]

    return [i.lower() for i in indicators]


def predict(i):
    tokenizer = Tokenizer(filters='\t\n', char_level=True, lower=True)
    word_dict_file = os.path.join(WORD_DICT)

    with open(word_dict_file) as F:
        txt = F.read()

    txt = json.loads(txt)
    tokenizer.word_index = txt

    i = normalize_domains(i)

    seq = tokenizer.texts_to_sequences(i)
    log_entry_processed = sequence.pad_sequences(seq, maxlen=MAX_STRING_LEN)

    MYMODEL = load_model(MODEL)
    MYMODEL.load_weights(WEIGHTS)
    p = MYMODEL.predict(log_entry_processed)
    K.clear_session()
    gc.collect()
    return p


def main():
    p = ArgumentParser(
        description=textwrap.dedent('''\
                example usage:
                    $ csirtg-domainsml-tf -i 'g00gle.com,google.com,paypal.com'
                    Using TensorFlow backend.
                    0.632489 - http://paypal.com
                    0.038573 - https://google.com/about-us
                    0.040402 - https://g0ogle.com/about-us
                '''),
        formatter_class=RawDescriptionHelpFormatter,
    )

    p.add_argument('-i', '--indicators', help='indicator(s), pipe delimited')
    p.add_argument('-d', '--debug', dest='debug', action="store_true")

    args = p.parse_args()

    if not sys.stdin.isatty():
        indicators = sys.stdin.read().split("\n")
        indicators = indicators[:-1]
    else:
        indicators = args.indicators.split(',')

    predictions = predict(indicators)

    for idx, v in enumerate(indicators):
        print("%f - %s" % (predictions[idx], v))


if __name__ == '__main__':
    main()
