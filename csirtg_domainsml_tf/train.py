#!/usr/bin/env python

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import textwrap
import json
import pandas
import optparse
import sys

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from csirtg_domainsml_tf.constants import MAX_STRING_LEN, MODEL, WEIGHTS, BATCH_SIZE, WORD_DICT

BATCH_SIZE = int(BATCH_SIZE)
MAX_STRING_LEN = int(MAX_STRING_LEN)

NEURONS = os.getenv('NEURONS', 16)
EMBEDDED_DIM = os.getenv('EMBEDDED_DIM', 32)

NEURONS = int(NEURONS)
EMBEDDED_DIM = int(EMBEDDED_DIM)

EPOCHS = os.getenv('EPOCHS', 3)
EPOCHS = int(EPOCHS)

# training split
SPLIT = os.getenv('TRAINING_SPLIT', .30)
SPLIT = float(SPLIT)

DATA_PATH = 'data'


def train(csv_file):
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='"', header=None)
    dataset = dataframe.sample(frac=1).values

    # Preprocess dataset
    X = dataset[:, 0]
    Y = dataset[:, 1]

    for index, item in enumerate(X):
        X[index] = item

    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)

    # Extract and save word dictionary
    word_dict_file = WORD_DICT

    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))

    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

    num_words = len(tokenizer.word_index)+1
    X = tokenizer.texts_to_sequences(X)

    max_log_length = MAX_STRING_LEN
    train_size = int(len(dataset) * (1 - SPLIT))

    X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
    X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    model = Sequential()
    model.add(Embedding(num_words, EMBEDDED_DIM, input_length=max_log_length))
    model.add(Dropout(0.5))
    model.add(LSTM(NEURONS, recurrent_dropout=0.5))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, validation_split=SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate model
    score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=BATCH_SIZE)

    print("Model Accuracy: {:0.2f}%".format(acc * 100))

    # Save model
    print("Saving model to: %s" % MODEL)
    model.save_weights(WEIGHTS)
    model.save(MODEL)


def main():
    p = ArgumentParser(
        description=textwrap.dedent('''\
            example usage:
                $ mkdir tmp
                $ cat data/whitelist.txt | csirtg-domainsml-tf-train --build --good > tmp/good.csv
                $ cat data/blacklist.txt | csirtg-domainsml-tf-train --build > tmp/bad.csv
                $ cat tmp/good.csv tmp/bad.csv | gshuf > data/training.csv
                $
                $ csirtg-domainsml-tf-train --training data/training.csv  # could take a few minutes to a few hours
            '''),
        formatter_class=RawDescriptionHelpFormatter,
    )

    p.add_argument('--good', action="store_true", default=False)
    p.add_argument('--build', action="store_true", help="Run in Build Mode (eg: build training data from "
                                                        "black/whitelists", default=False)

    p.add_argument('--training', help='path to training data [default: %(default)s]',
                   default=os.path.join('data', 'training.csv'))
    p.add_argument('-d', '--debug', dest='debug', action="store_true")

    args = p.parse_args()

    if args.build:
        for l in sys.stdin:
            l = l.rstrip()
            url = urlparse(l.lower())

            if args.good:
                print('%s,0' % url.geturl())
            else:
                print('%s,1' % url.geturl())

        raise SystemExit

    train(args.training)


if __name__ == '__main__':
    main()
