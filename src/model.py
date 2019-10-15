import mxnet as mx, gluonnlp as nlp
from mxnet.gluon import nn, rnn
from mxnet import nd, autograd, gluon
from bert.embedding import BertEmbedding
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
            # emb_a: batch_size*seq_len_a*emb_size, emb_b: batch_size*seq_len_b*emb_size
            # self.W: emb_size*emb_size
            # After the evaluation, the shape is batch_size*seq_len_a*emb_size_b
            self.op = nn.Lambda(lambda emb_a, emb_b: nd.softmax(
                nd.batch_dot(
                    nd.dot(emb_a, self.W),
                    nd.transpose(emb_b, axes=(0, 2, 1))
                ), axis=1)
            )

    def forward(self, emb_a, emb_b):
        return self.op(emb_a, emb_b)
        
class SoftAlignment(nn.Block):
    '''
    implement E_a/b = G_ab emb_a/(G_ab)^T emb_b
    '''
    def __init__(self, emb_size, **kwargs):
        super(SoftAlignment, self).__init__(**kwargs)
        with self.name_scope():
            # the parameter matrix W
            self.W = nd.random.normal(shape=(emb_size, emb_size))
            self.softalign = nn.Sequential(
                nn.Lambda(lambda G, H: nd.dot(
                    G, nd.transpose(H, axes=(1, 0, 2)))
                ),
                nn.Lambda(lambda E: nd.relu(nd.dot(E, self.W)))
            )

    def forward(self, G, H):
        return self.softalign(G, H) 

class BidirMatchEmb(nn.Block):
    '''
    Core model component
    '''
    def __init__(self, vocab_size, emb_size, nhidden, nlayers, dropout, **kwargs):
        '''
        init function, we will leave the bert embedding for now, suppose embeddings
        are already available
        '''
        super(BidirMatchEmb, self).__init__(**kwargs)
        with self.name_scope():
            # self.embedding_layer = BertEmbedding # FIXME: this is incomplete
            self.attweightmat = AttentionWeightMatrix(emb_size)
            self.softalign = SoftAlignment(emb_size)
            self.output = nn.Sequential(
                
            )

    def forward(self, emb_a, emb_b):
        raise NotImplementedError
