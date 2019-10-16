# !/usr/bin/env python3
from preprocess import get_dataloader
from model import DMCN
from train import train_valid
from mxnet import gluon
import logging, argparse
from model import ctx
from mxnet import init

parser = argparse.ArgumentParser(description='Train DMCN model')
parser.add_argument('--train_sentences', type=str, default='data/train.jsonl', help='Training set \
                    observations and hypotheses')
parser.add_argument('--train_labels', type=str, default='data/train-labels', help='Training set labels')
parser.add_argument('--test_sentences', type=str, default='data/dev.jsonl', help='Test set observations and hypotheses')
parser.add_argument('--test_labels', type=str, default='data/dev-labels', help='Test set labels')

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    # dataloader_train = get_dataloader(args.train_sentences, args.train_labels)
    dataloader_test = get_dataloader(args.test_sentences, args.test_labels)
    model = DMCN(768)
    model.initialize(init=init.Uniform(scale=1), ctx=ctx)
    loss_func = loss = gluon.loss.SoftmaxCrossEntropyLoss()
    lr, clip = .001, 2.5
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'clip_gradient': clip})
    train_valid(dataloader_test, dataloader_test, model, loss_func, trainer, num_epoch=3)
