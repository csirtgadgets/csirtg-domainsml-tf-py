# -*- coding: utf-8 -*-
from csirtg_domainsml_tf import predict
from faker import Faker
fake = Faker()
from pprint import pprint
import os
import numpy as np

DOMAINS = [
    'google.com',
    'g00gle.com',
    'aws.amazon.com',
    'ringcentral.com',
    'security.duke.edu',
    'gallery.mailchimp.com',
    'csirtg.io',
    'bankwest.com.au',
    'bank.wellsbankingsecurelogin.com',
    'apple-gps-tracker.xyz'
]

THRESHOLD = 0.92
SAMPLE = int(os.getenv('CSIRTG_DOMAINSML_TEST_SAMPLE', 200))


def test_domains_basic():
    predictions = predict(DOMAINS)

    assert predictions[0]

    assert np.average(predictions) > 0.5


def test_random():
    s = []
    for d in range(0, SAMPLE):
        s.append(str(fake.uri()))

    predictions = predict(s)

    assert predictions[0]

    assert np.average(predictions) > 0.7
