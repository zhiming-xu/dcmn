# !/usr/bin/env python3
import json, logging
import mxnet as mx
from mxnet import nd
import multiprocessing as mp
from collections import defaultdict
from bert.embedding import BertEmbedding

logger = logging.Logger(__name__)
logger.setLevel(logging.WARNING)

def load_jsonl(filepath):
    '''
    this function will load a *.jsonl file by the name of filepath,
    note: each line of this file is a dict {}, not standard json
    '''
    logging.info('Load jsonl file from {}'.format(filepath))
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    return data

def load_sentences(filepath, keys=['obs1', 'obs2', 'hyp1', 'hyp2']):
    '''
    this function will use load_jsonl to load the jsonl file, extract
    the observations and hypotheses, return: key -> list of strings
    '''
    raw_data = load_jsonl(filepath)
    ret = defaultdict(list)
    logger.info('Build the dictionary for each kind of sentences')
    [[ret[key].append(entry[key]) for key in keys] for entry in raw_data]
    return ret

def to_dataloader(sentences, ctx=mx.gpu(), batch_size=32, max_seq_length=20):
    '''
    this function will use BertEmbedding to get dataloader for each
    kind of sentences
    '''
    bertembedding = BertEmbedding(ctx=ctx, batch_size=batch_size, max_seq_length=max_seq_length)
    logger.info('Construct bert embedding for these fields')
    for sts in sentences:
        tokens_embs = bertembedding.embedding(sts)
        embs = [nd.array(token_emb[1]) for token_emb in tokens_embs]



