import mxnet as mx, gluonnlp as nlp
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd, gluon
import pandas as pd, numpy as np

np.random.seed(9102)
mx.random.seed(8064)

class AttentionWeightMatrix(nn.Block):
    '''
    implement G^qa = softmax(H^p W H^aT)
    '''
    def __init__(self, embsize, **kwargs):
        super(AttentionWeightMatrix, self).__init__(**kwargs)
        with self.name_scope():
            self.W = nd.normal(shape=(embsize, embsize))
            # emb_a: batch_size*seq_len*emb_size, emb_b: batch_size*seq_len*emb_size
            # self.W: emb_size*emb_size
            # After the evaluation, the shape is batch_size*emb_size*emb_size
            self.op = nn.Lambda(lambda emb_a, emb_b: nd.batch_dot(
                nd.dot(emb_a, self.W),
                nd.transpose(emb_b, axes=(0, 2, 1)))
            )

    def forward(self, emb_a, emb_b):
        return self.op(emb_a, emb_b)
        

class DualCoMatchNetwork(nn.Block):
    '''
    Core model component
    '''
    def __init__(self, vocab_size, embsize, nhidden, nlayers, dropout, **kwargs):
        '''
        init function, we will leave the bert embedding for now, suppose embeddings
        are already available
        '''
        super(DualCoMatchNetwork, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = dropout
            self.embedding_layer = nn.Embedding(vocab_size, embsize)
            self.W = nd.normal(shape=(embsize, embsize))