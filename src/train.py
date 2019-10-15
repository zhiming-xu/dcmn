# !/usr/bin/env python3
import mxnet as mx
from mxnet import gluon, autograd
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)
