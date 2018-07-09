#!/bin/bash

set -e

rm -rf tmp
mkdir tmp

cat data/whitelist.txt | csirtg-domainsml-tf-train --build --good > tmp/whitelist.csv
cat data/blacklist.txt | csirtg-domainsml-tf-train --build > tmp/blacklist.csv
cat tmp/whitelist.csv tmp/blacklist.csv | gshuf > data/training.csv
time csirtg-domainsml-tf-train
