# !/usr/bin/env python3
# helpers for training the model
import mxnet as mx
from mxnet import autograd, nd
import logging, time
import numpy as np

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.Logger(__name__)
logger.setLevel(logging.WARNING)


def calculate_loss(inputs, labels, model, loss_func, loss_name='sce', class_weight=None):
    '''
    this function will calculate loss, we use softmax cross entropy for now,
    possibly extend to weighted version
    '''
    preds = model(inputs)
    if loss_name == 'sce':
        l = loss_func(preds, labels)
    else:
        logger.error('Loss function %s not implemented' % loss_name)
        raise NotImplementedError
    return preds, l

def one_epoch(dataloader, model, loss_func, trainer, ctx, is_train, epoch,
              class_weight=None, loss_name='sce'):
    '''
    this function trains model for one epoch if `is_train` is True
    also calculates loss/metrics whether in training or dev
    '''
    loss_val = 0.
    metric = mx.metric.Accuracy()

    for n_batch, iters in enumerate(dataloader):
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
        batch_pred = nd.softmax(batch_pred, axis=1)
        batch_true = labels.reshape(-1)
        metric.update(preds=batch_pred, labels=batch_true)

        batch_loss = l.mean().asscalar()
        loss_val += batch_loss

    # metric
    loss_val /= n_batch + 1

    if is_train:
        print('epoch %d, learning_rate %.5f \n\t train_loss %.4f, %s: %.4f' %
             (epoch, trainer.learning_rate, loss_val, *metric.get()))
        # train_curve.append((acc, F1))
        # declay lr
        if epoch % 2 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)
    else:
        print('\t valid_loss %.4f, %s: %.4f' % (loss_val, *metric.get()))
        # valid_curve.append((acc, F1))

def train_valid(dataloader_train, dataloader_test, model, loss_func, trainer, \
                num_epoch, ctx, class_weight=None, loss_name='sce'):
    '''
    wrapper for training and "test" the model
    '''
    for epoch in range(1, num_epoch+1):
        start = time.time()
        # train
        is_train = True
        one_epoch(dataloader_train, model, loss_func, trainer, ctx, is_train, \
                  epoch, class_weight, loss_name)

        # valid
        is_train = False
        one_epoch(dataloader_test, model, loss_func, trainer, ctx, is_train, \
                  epoch, class_weight, loss_name)
        end = time.time()
        print('time %.2f sec' % (end-start))
        print("*"*100)

def inference(model, sample):
    '''
    do inference for one sample
    '''
    for i, embs in enumerate(samples):
        pred = model(embs).argmax(axis=-1).astype('int32').asscalar()
        # samples[pred+2]: hyp[pred], [i], hyp[pred] for i th sample
        print('Sample:\034[32m', list(zip(*samples))[i], '\033[0m\nPred:\t\033[33m', \
              samples[pred+2][i], '\033[0m')