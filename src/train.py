# !/usr/bin/env python3
# helpers for training the model
import mxnet as mx
from mxnet import autograd, nd
from sklearn.metrics import accuracy_score, f1_score
import logging, time
import numpy as np

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

logger.info('This module will try to use GPU as default. If it is not available, will \
            switch to CPU')

try:
    nd.ones(10, ctx=mx.gpu())
    ctx = mx.gpu()
except Exception as e:
    logger.warning('No GPU device found with exception {}. Will use CPU instead'.format(e))
    ctx = mx.cpu()

def calculate_loss(inputs, labels, model, loss_func, loss_name='sce', class_weight=None):
    '''
    this function will calculate loss, we use softmax cross entropy for now,
    possibly extend to weighted version
    '''
    preds = model(inputs)
    labels = nd.array(labels.astype('int32', copy=False), ctx=ctx)
    if loss_name == 'sce':
        l = loss_func(preds, labels)
    elif loss_name == 'wsce':
        l = loss_func(preds, labels, class_weight, class_weight.shape[0])
    else:
        raise NotImplementedError
    return preds, l

def one_epoch(dataloader, model, loss_func, trainer, ctx, is_train, epoch, class_weight=None, loss_name='sce'):
    '''
    this function trains model for one epoch if `is_train` is True
    also calculates loss/metrics whether in training or dev
    '''
    loss_val = 0.
    total_pred = []
    total_true = []
    n_batch = 0

    for iters in dataloader:
        *inputs, labels = [item.as_in_context(ctx) for item in iters]

        if is_train:
            with autograd.record():
                batch_pred, l = calculate_loss(inputs, labels, model, loss_func, loss_name, class_weight)

            # backward calculate
            l.backward()

            # update parmas
            trainer.step(labels.shape[0])

        else:
            batch_pred, l = calculate_loss(inputs, labels, model, loss_func, loss_name, class_weight)

        # keep result for metric
        batch_pred = nd.argmax(nd.softmax(batch_pred, axis=1), axis=1).asnumpy()
        batch_true = np.reshape(labels.asnumpy(), (-1, ))
        total_pred.extend(batch_pred.tolist())
        total_true.extend(batch_true.tolist())
        
        batch_loss = l.mean().asscalar()

        n_batch += 1
        loss_val += batch_loss

        # check the result of traing phase
        if is_train and n_batch % 200 == 0:
            logger.info('epoch %d, batch %d, batch_train_loss %.4f, batch_train_acc %.3f' %
                       (epoch, n_batch, batch_loss, accuracy_score(batch_true, batch_pred)))

    # metric
    F1 = f1_score(np.array(total_true), np.array(total_pred), average='binary')
    acc = accuracy_score(np.array(total_true), np.array(total_pred))
    loss_val /= n_batch

    if is_train:
        logger.info('epoch %d, learning_rate %.5f \n\t train_loss %.4f, acc_train %.3f, F1_train %.3f, ' %
                   (epoch, trainer.learning_rate, loss_val, acc, F1))
        # train_curve.append((acc, F1))
        # declay lr
        if epoch % 2 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)
    else:
        logger.info('\t valid_loss %.4f, acc_valid %.3f, F1_valid %.3f, ' % (loss_val, acc, F1))
        # valid_curve.append((acc, F1))

def train_valid(dataloader_train, dataloader_test, model, loss_func, trainer, \
                ctx, num_epoch, class_weight=None, loss_name='sce'):
    '''
    wrapper for training and "test" the model
    '''
    for epoch in range(1, num_epoch+1):
        start = time.time()
        # train
        is_train = True
        one_epoch(dataloader_train, model, loss_func, trainer, ctx, is_train, epoch, class_weight, loss_name)

        # valid
        is_train = False
        one_epoch(dataloader_test, model, loss_func, trainer, ctx, is_train, epoch, class_weight, loss_name)
        end = time.time()
        logger.info('time %.2f sec' % (end-start))
        logger.info("*"*100)