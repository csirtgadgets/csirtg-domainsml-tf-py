# csirtg-domainsml-tf-py
simple python/keras/tensorflow library for detecting odd domains in python

# References

https://csirtgadgets.com/commits/2018/7/7/how-to-hunt-for-phishing-domains-using-tensorflow
https://github.com/csirtgadgets/tf-domains-example
https://github.com/csirtgadgets/csirtg-urlsml-tf-py

https://medium.com/slalom-engineering/detecting-malicious-requests-with-keras-tensorflow-5d5db06b4f28
https://csirtgadgets.com/commits/2018/3/8/hunting-for-suspicious-domains-using-python-and-sklearn
https://csirtgadgets.com/commits/2018/3/30/hunting-for-threats-like-a-quant

# Getting Started

## Incorporating into a Project

```bash
$ pip install csirtg_domainsml_tf
```

```python
from csirtg_domainsml_tf import predict
from pprint import pprint

indicators = ['google.com', 'g0gole.com', 'paypal.com', 'safe-paypal-trustme.com']
predictions = predict(indicators)

for idx, v in enumerate(indicators):
    print("%f - %s" % (predictions[idx], v))
```


## Development and Building
```bash
$ pip install -r dev_requirements.txt
$ python setup.py develop

$ csirtg-domainsml-tf -i google.com,apple.com,paypal.com,paypal-ate-my-lunch.com,google-analytics.com,securitymywindowspcsystem.info,bank.wellsbankingsecurelogin.com,apple-gps-tracker.xyz
Using TensorFlow backend.
0.241244 - google.com
0.307995 - apple.com
0.448235 - paypal.com
0.777552 - paypal-ate-my-lunch.com
0.605344 - google-analytics.com
0.783592 - securitymywindowspcsystem.info
0.819034 - bank.wellsbankingsecurelogin.com
0.838330 - apple-gps-tracker.xyz
```

# Rebuilding Models

If you want to rebuild the models with your own data:

1. Update `data/whitelist.txt`
1. Update `data/blacklist.txt`
1. Run the `helpers/build.sh` command

```bash
$ bash helpers/build.sh  # this will take a few minutes...

Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 255, 32)           1280
_________________________________________________________________
dropout_1 (Dropout)          (None, 255, 32)           0
_________________________________________________________________
lstm_1 (LSTM)                (None, 16)                3136
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17
=================================================================
Total params: 4,433
Trainable params: 4,433
Non-trainable params: 0
_________________________________________________________________
None
Train on 23642 samples, validate on 10133 samples
Epoch 1/3
23642/23642 [==============================] - 24s 1ms/step - loss: 0.6750 - acc: 0.6192 - val_loss: 0.6328 - val_acc: 0.6762
Epoch 2/3
23642/23642 [==============================] - 23s 979us/step - loss: 0.6067 - acc: 0.6824 - val_loss: 0.5474 - val_acc: 0.7369
Epoch 3/3
23642/23642 [==============================] - 23s 994us/step - loss: 0.5451 - acc: 0.7392 - val_loss: 0.4626 - val_acc: 0.7958
14475/14475 [==============================] - 2s 168us/step
Model Accuracy: 79.45%

```
