# !/usr/bin/env python3
import logging
import mxnet as mx
from mxnet import nd, gluon
import multiprocessing as mp
import gluonnlp as nlp
import numpy as np
from collections import defaultdict
from bert.embedding import BertEmbedding
from util import load_labels, load_sentences, _get_threads

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')
logger = logging.Logger(__name__)
logger.setLevel(logging.WARNING)

def to_dataset(samples, labels, ctx=mx.gpu(), batch_size=64, max_seq_length=25):
    '''
    this function will use BertEmbedding to get each fields' embeddings
    and load the given labels, put them together into a dataset
    '''
    bertembedding = BertEmbedding(ctx=ctx, batch_size=batch_size, max_seq_length=max_seq_length)
    logger.info('Construct bert embedding for sentences')

    embs = []
    for sample in samples:
        tokens_embs = bertembedding.embedding(sample)
        embs.append([np.asarray(token_emb[1]) for token_emb in tokens_embs])
    if labels: 
        dataset = [[*obs_hyp, label] for obs_hyp, label in zip(embs, labels)]
    else:
        dataset = embs
    return dataset

def get_length(dataset):
    '''
    lengths used for batch sampler, we will use the first field of each row
    for now, i.e., obs1
    '''
    return [row[0].shape[0] for row in dataset]

def to_dataloader(dataset, batch_size=64, num_buckets=10, bucket_ratio=.5):
    '''
    this function will sample the dataset to dataloader
    '''
    pads = [nlp.data.batchify.Pad(axis=0, pad_val=0) for _ in range(len(dataset[0])-1)]
    batchify_fn = nlp.data.batchify.Tuple(
        *pads,                      # for observations and hypotheses
        nlp.data.batchify.Stack()   # for labels
    )
    lengths = get_length(dataset)

    print('Build batch_sampler')
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        lengths=lengths, batch_size=batch_size, num_buckets=num_buckets,
        ratio=bucket_ratio, shuffle=True
    )
    print(batch_sampler.stats())

    dataloader = gluon.data.DataLoader(
        dataset, batch_sampler=batch_sampler, batchify_fn=batchify_fn,
        num_workers=_get_threads()
    )
    
    return dataloader

def get_dataloader(sts, labels=None, keys=['obs1', 'obs2', 'hyp1', 'hyp2'], \
                   batch_size=64, num_buckets=10, bucket_ratio=.5, \
                   ctx=mx.gpu(), max_seq_length=25, sample_num=None):
    '''
    this function will use the helpers above, take sentence file path,
    label file path, and batch_size, num_buckets, bucket_ratio, to
    get the dataloader for model to us. sample_num controls how many
    samples in dataset the model will use, defualt to None, e.g., use all
    '''
    if labels:
        sentences = load_sentences(sts, keys=keys)
        sentences = sentences[:sample_num]
        labels = load_labels(labels)[:sample_num]
        
        try:
            assert(len(sentences)==len(labels))
        except:
            logger.error('Sample sentence length does not equal to label\'s length!')
            exit(-1)

        dataset = to_dataset(sentences, labels, ctx=ctx, batch_size=batch_size, \
                             max_seq_length=max_seq_length)

        dataloader = to_dataloader(dataset=dataset, batch_size=batch_size, \
                                   num_buckets=num_buckets, bucket_ratio=bucket_ratio)
    else:
        dataset = to_dataset(sts, labels, ctx=ctx, batch_size=batch_size, \
                             max_seq_length=max_seq_length)
        dataloader = []
        for sample in dataset:
            batch = []
            for emb in sample:
                batch.append(nd.array(emb.reshape(1, *emb.shape)))
            dataloader.append(batch)

    return dataloader