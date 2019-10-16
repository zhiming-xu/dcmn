# !/usr/bin/env python3
from preprocess import get_dataloader
from model import DMCN
from train import train_valid
import logging, argparse

parser = argparse.ArgumentParser(description='Train DMCN model')
parser.add_argument('--train_sentences', type=str, default='data/train.jsonl', help='Training set \
                    observations and hypotheses')
parser.add_argument('--train_lables', type=str, default='data/train-labels', help='Training set labels')
parser.add_argument('--test_sentences', type=str, default='data/test.jsonl', help='Test set observations and hypotheses')
parser.add_argument('--test_lables', type=str, default='data/test-labels', help='Test set labels')

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

if __name__ == '__main__':
    dataloader_train = get_dataloader(args.train_sentences, args.train_labels)
    dataloader_test = get_dataloader(args.test_sentences, args.test_labels)
