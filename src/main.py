# !/usr/bin/env python3
import preprocess, model, train
import mxnet as mx
from mxnet import gluon, init
import logging, argparse

parser = argparse.ArgumentParser(description='Train DMCN model')
parser.add_argument('--train_sentences', type=str, default='data/train.jsonl', help='Training set \
                    observations and hypotheses')
parser.add_argument('--train_labels', type=str, default='data/train-labels', help='Training set labels')
parser.add_argument('--test_sentences', type=str, default='data/dev.jsonl', help='Test set observations and hypotheses')
parser.add_argument('--test_labels', type=str, default='data/dev-labels', help='Test set labels')
parser.add_argument('--inference', type=int, default=0, \
                    help='inference with exsiting model, default to 0/False')
parser.add_argument('--model_params', type=str, default='results/dmcn.params', \
                    help='path to model params for inference')
parser.add_argument('--sample', type=str, help='observations and hypotheses for inference, \
                    each sentence should be separated with |')

args = parser.parse_args()

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def inference(model, samples):
    '''
    do inference for a list of samples, in the form of [[obs1's], [obs2's], [hyp1's], [hyp2's], ...]
    '''
    dataloader = preprocess.get_dataloader(samples)
    for i, embs in enumerate(dataloader):
        # embs is [emb(obs1), emb(obs2), emb(hyp1), emb(hyp2), ...]
        pred = model(embs).argmax(axis=-1).astype('int32').asscalar()
        # samples[pred+2]: hyp[pred], [i], hyp[pred] for i th sample
        print('\033[36mSamples:\033[34m')
        for sentence in list(zip(*samples))[i]:
            print(sentence)
        print('\033[36mPred:\033[32m')
        print(samples[pred+2][i], '\033[0m')
        # print('\033[36mSample:\033[34m', list(zip(*samples))[i], \
        #       '\033[36m\nPred:\t\033[32m', samples[pred+2][i], '\033[0m')

if __name__ == '__main__':
    if args.inference:
        # do inference
        dcmn = model.DCMN()
        dcmn.load_parameters(args.model_params)
        sts = args.sample.split('|')
        samples = [[sentence.strip()] for sentence in sts]
        inference(dcmn, samples)
    else:
        # do training
        dataloader_train = preprocess.get_dataloader(
            sts=args.train_sentences, labels=args.train_labels, sample_num=6000
        )
        dataloader_test = preprocess.get_dataloader(
            sts=args.test_sentences, labels=args.test_labels
        )
        dcmn = model.DCMN()
        dcmn.initialize(init=init.Uniform(.03), ctx=mx.gpu())
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        lr, clip = .001, 2.5
        trainer = gluon.Trainer(dcmn.collect_params(), 'adam',
                                {'learning_rate': lr, 'clip_gradient': clip})
        train.train_valid(dataloader_train, dataloader_test, dcmn, loss_func,
                        trainer, num_epoch=15, ctx=mx.gpu())
        dcmn.save_parameters('dmcn15.params')
